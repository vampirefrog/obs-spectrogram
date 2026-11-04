/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "spectrogram-source.h"
#include "fft.h"

#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/bmem.h>
#include <util/platform.h>
#include <util/dstr.h>

#include <util/threading.h>
#include <media-io/audio-io.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define T(key) obs_module_text(key)

/* ── limits ─────────────────────────────────────────────────────────────── */

#define MAX_FFT_SIZE      8192
#define MAX_HEIGHT        8192
#define MAX_PALETTE       16
#define RING_SIZE         (MAX_FFT_SIZE * 8)
#define MAX_RESONATE_BINS 1024

/* Reserved audio-source setting value: capture the final output mix
   (Track 1) instead of a single source. */
#define OUTPUT_MIX_ID     "__output_mix__"
#define OUTPUT_MIX_TRACK  0

/* how often to retry finding an audio source that isn't there (yet) */
#define RECONNECT_INTERVAL 1.0f

/* ── analysis modes ──────────────────────────────────────────────────────── */

#define ANALYSIS_FFT      0
#define ANALYSIS_RESONATE 1

/* ── window types ────────────────────────────────────────────────────────── */

#define WIN_RECT     0
#define WIN_HANN     1
#define WIN_HAMMING  2
#define WIN_BLACKMAN 3

/* ── settings keys ───────────────────────────────────────────────────────── */

#define S_AUDIO_SRC       "audio_source"
#define S_WIDTH           "width"
#define S_HEIGHT          "height"
#define S_ANALYSIS_MODE   "analysis_mode"
/* FFT-mode settings */
#define S_FFT_SIZE        "fft_size"
#define S_WINDOW          "window_type"
#define S_OVERLAP         "overlap"
/* Resonate-mode settings */
#define S_RESONATE_BINS   "resonate_bins"
#define S_RESONATE_HOP    "resonate_hop"
#define S_RESONATE_SMOOTH "resonate_smooth"
/* common */
#define S_LOG_SCALE       "log_scale"
#define S_MIN_FREQ        "min_freq"
#define S_MAX_FREQ        "max_freq"
#define S_DB_MIN          "db_min"
#define S_DB_MAX          "db_max"
#define S_GAIN            "gain_db"
#define S_COLOR_0         "color_0"
#define S_COLOR_1         "color_1"
#define S_COLOR_2         "color_2"

/* ── context ─────────────────────────────────────────────────────────────── */

struct spectrogram_source {
	obs_source_t *source;

	/* audio input: audio_ref is the setting value (source UUID, a legacy
	   source name, or OUTPUT_MIX_ID); at most one of audio_weak /
	   output_mix_connected is live at a time */
	char              *audio_ref;
	obs_weak_source_t *audio_weak;
	volatile bool      audio_removed;  /* set by the source's "remove" signal */
	bool               output_mix_connected;
	float              reconnect_timer;

	/* display */
	uint32_t width;
	uint32_t height;

	/* analysis mode */
	int      analysis_mode;

	/* FFT / signal settings */
	uint32_t fft_size;
	uint32_t hop_size;
	int      window_type;

	/* Resonate settings */
	uint32_t resonate_bins;
	uint32_t resonate_hop;
	bool     resonate_smooth;

	/* common signal settings */
	bool     log_scale;
	float    min_freq;
	float    max_freq;
	float    db_min;
	float    db_max;
	float    gain_db;

	/* palette (ARGB: 0xAARRGGBB, compatible with GS_BGRA uint32_t) */
	uint32_t palette[MAX_PALETTE];
	int      palette_count;

	/* precomputed FFT window coefficients */
	float    win[MAX_FFT_SIZE];

	/* audio ring buffer – audio thread writes, render thread reads */
	float    ring[RING_SIZE];
	uint32_t ring_r;
	uint32_t ring_w;
	uint32_t ring_n;
	uint32_t sample_rate;
	uint32_t channels;

	/* FFT scratch (render thread only) */
	float    fft_re[MAX_FFT_SIZE];
	float    fft_im[MAX_FFT_SIZE];

	/* ── Resonate bank state ──────────────────────────────────────────── *
	 * Resonator k is described by Francois (ICMC 2025, "Resonate").      *
	 * Per-sample update (Eq. 5):                                          *
	 *   P ← P · pstep_k          (rotate phasor by e^{-iω_k·Δt})        *
	 *   R ← (1−α_k)·R + α_k·x·P  (EWMA of demodulated signal)           *
	 * Optional second smoothing (Eq. 8):                                  *
	 *   R̃ ← (1−α_k)·R̃ + α_k·R                                          *
	 * Frequency-dependent alpha heuristic (Eq. 7):                        *
	 *   α_f = 1 − e^{−Δt·f / log(1+f)}                                   *
	 * ────────────────────────────────────────────────────────────────── */
	float    res_freq    [MAX_RESONATE_BINS]; /* resonant frequency      */
	float    res_alpha   [MAX_RESONATE_BINS]; /* per-resonator α_f       */
	float    res_pstep_re[MAX_RESONATE_BINS]; /* Re(e^{-iω_k·Δt})        */
	float    res_pstep_im[MAX_RESONATE_BINS]; /* Im(e^{-iω_k·Δt})        */
	float    res_p_re    [MAX_RESONATE_BINS]; /* phasor P: real part      */
	float    res_p_im    [MAX_RESONATE_BINS]; /* phasor P: imaginary part */
	float    res_r_re    [MAX_RESONATE_BINS]; /* resonator R: real        */
	float    res_r_im    [MAX_RESONATE_BINS]; /* resonator R: imaginary   */
	float    res_rt_re   [MAX_RESONATE_BINS]; /* smoothed R̃: real         */
	float    res_rt_im   [MAX_RESONATE_BINS]; /* smoothed R̃: imaginary    */

	/* column produced by compute_column, consumed by write_column */
	uint32_t col_pixels[MAX_HEIGHT];

	/* GPU texture */
	gs_texture_t *tex;
	uint32_t     *tex_cpu;
	uint32_t      write_col;

	/* shader */
	gs_effect_t *effect;
	gs_eparam_t *ep_image;

	pthread_mutex_t mutex;
};

/* ── helpers ─────────────────────────────────────────────────────────────── */

static void build_window(float *out, uint32_t n, int type)
{
	for (uint32_t i = 0; i < n; i++) {
		float t = (float)i / (float)(n - 1);
		float w;
		switch (type) {
		case WIN_HANN:
			w = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * t));
			break;
		case WIN_HAMMING:
			w = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * t);
			break;
		case WIN_BLACKMAN:
			w = 0.42f
			  - 0.5f  * cosf(2.0f * (float)M_PI * t)
			  + 0.08f * cosf(4.0f * (float)M_PI * t);
			break;
		default:
			w = 1.0f;
			break;
		}
		out[i] = w;
	}
}

/* Initialise / reinitialise the Resonate resonator bank.
 * Computes geometrically spaced frequencies, per-resonator α from the
 * heuristic in Eq. 7 of the paper, and the phasor step e^{-iω_k·Δt}.
 * Resets all accumulator state to zero / unit phasor. */
static void init_resonators(struct spectrogram_source *ctx)
{
	uint32_t N  = ctx->resonate_bins;
	float    dt = 1.0f / (float)ctx->sample_rate;
	float    f0 = ctx->min_freq < 1.0f ? 1.0f : ctx->min_freq;
	float    f1 = ctx->max_freq;
	if (f1 <= f0) f1 = f0 + 1.0f;

	for (uint32_t k = 0; k < N; k++) {
		/* geometrically spaced (log-uniform) frequencies (Sec. 4.1) */
		float t = (N > 1) ? (float)k / (float)(N - 1) : 0.0f;
		float f = f0 * powf(f1 / f0, t);
		ctx->res_freq[k] = f;

		/* α_f = 1 − e^{−Δt·f / log(1+f)}   (Eq. 7) */
		ctx->res_alpha[k] = 1.0f - expf(-dt * f / logf(1.0f + f));

		/* phasor step: e^{−iω_k·Δt} = cos(ω_k·Δt) − i·sin(ω_k·Δt) */
		float omega = 2.0f * (float)M_PI * f;
		ctx->res_pstep_re[k] =  cosf(omega * dt);
		ctx->res_pstep_im[k] = -sinf(omega * dt);

		/* phasor starts at 1 + 0i */
		ctx->res_p_re[k] = 1.0f;
		ctx->res_p_im[k] = 0.0f;

		/* accumulators start at zero */
		ctx->res_r_re [k] = 0.0f;
		ctx->res_r_im [k] = 0.0f;
		ctx->res_rt_re[k] = 0.0f;
		ctx->res_rt_im[k] = 0.0f;
	}
}

/* Clamp a fractional array index to [0, n-1], mapping NaN to 0, so it
   can be split into an integer part and a fraction safely. */
static inline float clamp_index(float i, uint32_t n)
{
	if (!(i > 0.0f))
		return 0.0f;
	return i < (float)(n - 1) ? i : (float)(n - 1);
}

static inline uint8_t lerp_u8(uint8_t a, uint8_t b, float t)
{
	float v = (float)a + t * ((float)b - (float)a);
	return (uint8_t)(v < 0.f ? 0 : v > 255.f ? 255 : v);
}

static uint32_t palette_lookup(float db, float db_min, float db_max,
                               const uint32_t *pal, int count)
{
	float t = (db - db_min) / (db_max - db_min);
	t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);

	float  fi = t * (float)(count - 1);
	int    i0 = (int)fi;
	float  fr = fi - (float)i0;
	int    i1 = i0 + 1 < count ? i0 + 1 : i0;

	uint32_t c0 = pal[i0], c1 = pal[i1];
	uint8_t a = lerp_u8((c0>>24)&0xFF, (c1>>24)&0xFF, fr);
	uint8_t r = lerp_u8((c0>>16)&0xFF, (c1>>16)&0xFF, fr);
	uint8_t g = lerp_u8((c0>> 8)&0xFF, (c1>> 8)&0xFF, fr);
	uint8_t b = lerp_u8( c0     &0xFF,  c1     &0xFF,  fr);
	return ((uint32_t)a<<24)|((uint32_t)r<<16)|((uint32_t)g<<8)|b;
}

/* ── audio callback (audio thread) ──────────────────────────────────────── */

/* Downmix all channels of a planar-float block to mono and append it to
   the ring buffer.  Only the first ctx->channels planes are valid: the
   output-mix path leaves the rest of data->data[] uninitialised. */
static void push_audio_frames(struct spectrogram_source *ctx,
                              const struct audio_data *data)
{
	const float *planes[MAX_AV_PLANES];
	uint32_t     nch = 0;
	for (uint32_t c = 0; c < ctx->channels; c++) {
		if (data->data[c])
			planes[nch++] = (const float *)data->data[c];
	}
	if (!nch)
		return;

	float    scale = 1.0f / (float)nch;
	uint32_t nf    = data->frames;

	pthread_mutex_lock(&ctx->mutex);
	for (uint32_t i = 0; i < nf; i++) {
		float x = 0.0f;
		for (uint32_t c = 0; c < nch; c++)
			x += planes[c][i];
		ctx->ring[ctx->ring_w] = x * scale;
		ctx->ring_w = (ctx->ring_w + 1) % RING_SIZE;
		if (ctx->ring_n < RING_SIZE) {
			ctx->ring_n++;
		} else {
			ctx->ring_r = (ctx->ring_r + 1) % RING_SIZE;
		}
	}
	pthread_mutex_unlock(&ctx->mutex);
}

static void capture_source_audio(void *param, obs_source_t *source,
                                 const struct audio_data *data, bool muted)
{
	(void)source;
	if (!muted)
		push_audio_frames(param, data);
}

static void capture_output_mix(void *param, size_t mix_idx,
                               struct audio_data *data)
{
	(void)mix_idx;
	push_audio_frames(param, data);
}

/* ── FFT column (graphics thread) ────────────────────────────────────────── */

static bool compute_column_fft(struct spectrogram_source *ctx)
{
	pthread_mutex_lock(&ctx->mutex);
	if (ctx->ring_n < ctx->fft_size) {
		pthread_mutex_unlock(&ctx->mutex);
		return false;
	}

	for (uint32_t i = 0; i < ctx->fft_size; i++) {
		ctx->fft_re[i] = ctx->ring[(ctx->ring_r + i) % RING_SIZE];
		ctx->fft_im[i] = 0.0f;
	}
	ctx->ring_r = (ctx->ring_r + ctx->hop_size) % RING_SIZE;
	ctx->ring_n = ctx->ring_n >= ctx->hop_size ? ctx->ring_n - ctx->hop_size : 0;
	pthread_mutex_unlock(&ctx->mutex);

	for (uint32_t i = 0; i < ctx->fft_size; i++)
		ctx->fft_re[i] *= ctx->win[i];

	fft_forward(ctx->fft_re, ctx->fft_im, ctx->fft_size);

	uint32_t num_bins = ctx->fft_size / 2 + 1;
	float    norm     = 1.0f / (float)(ctx->fft_size / 2);
	for (uint32_t i = 0; i < num_bins; i++) {
		float re = ctx->fft_re[i];
		float im = ctx->fft_im[i];
		float m  = sqrtf(re*re + im*im) * norm;
		ctx->fft_im[i] = (m > 1e-12f)
			? 20.0f * log10f(m) + ctx->gain_db
			: ctx->db_min;
	}

	float sr2    = (float)ctx->sample_rate * 0.5f;
	float f_min  = ctx->min_freq < 1.0f ? 1.0f : ctx->min_freq;
	float f_max  = ctx->max_freq > sr2   ? sr2  : ctx->max_freq;
	float log_lo = logf(f_min);
	float log_hi = logf(f_max);

	for (uint32_t y = 0; y < ctx->height; y++) {
		float t = 1.0f - (float)y / (float)(ctx->height - 1);
		float freq = ctx->log_scale
			? expf(log_lo + t * (log_hi - log_lo))
			: f_min + t * (f_max - f_min);

		float bin_f = clamp_index(freq * (float)ctx->fft_size / (float)ctx->sample_rate,
		                          num_bins);
		int   b0    = (int)bin_f;
		float frac  = bin_f - (float)b0;
		int   b1    = b0 + 1 < (int)num_bins ? b0 + 1 : b0;

		float db = ctx->fft_im[b0] + frac * (ctx->fft_im[b1] - ctx->fft_im[b0]);
		ctx->col_pixels[y] = palette_lookup(db, ctx->db_min, ctx->db_max,
		                                    ctx->palette, ctx->palette_count);
	}

	return true;
}

/* ── Resonate column (graphics thread) ───────────────────────────────────── *
 *                                                                             *
 * Implements "Resonate: Efficient Low Latency Spectral Analysis of Audio     *
 * Signals" by Alexandre R. J. François, ICMC 2025.                           *
 *                                                                             *
 * Drains resonate_hop samples from the ring buffer.  For each resonator k    *
 * the inner loop runs sample-by-sample so the 6 accumulator values stay in   *
 * registers across the entire hop (better cache behaviour than the           *
 * sample-outer / resonator-inner ordering).                                   *
 * ─────────────────────────────────────────────────────────────────────────── */

static bool compute_column_resonate(struct spectrogram_source *ctx)
{
	uint32_t hop = ctx->resonate_hop;

	pthread_mutex_lock(&ctx->mutex);
	if (ctx->ring_n < hop) {
		pthread_mutex_unlock(&ctx->mutex);
		return false;
	}

	/* copy hop samples into the fft_re scratch buffer so the lock can be
	   released before the (potentially long) resonator update loop */
	for (uint32_t s = 0; s < hop; s++) {
		ctx->fft_re[s] = ctx->ring[ctx->ring_r];
		ctx->ring_r = (ctx->ring_r + 1) % RING_SIZE;
	}
	ctx->ring_n -= hop;
	pthread_mutex_unlock(&ctx->mutex);

	const float *samples = ctx->fft_re;

	/* update each resonator across all hop samples */
	for (uint32_t k = 0; k < ctx->resonate_bins; k++) {
		float pr  = ctx->res_p_re[k];
		float pi  = ctx->res_p_im[k];
		float rr  = ctx->res_r_re[k];
		float ri  = ctx->res_r_im[k];
		float rtr = ctx->res_rt_re[k];
		float rti = ctx->res_rt_im[k];

		float psr = ctx->res_pstep_re[k];
		float psi = ctx->res_pstep_im[k];
		float a   = ctx->res_alpha[k];
		float a1  = 1.0f - a;

		for (uint32_t s = 0; s < hop; s++) {
			float x = samples[s];

			/* P ← P · pstep  (Eq. 5, first line) */
			float new_pr = pr * psr - pi * psi;
			float new_pi = pr * psi + pi * psr;
			pr = new_pr;
			pi = new_pi;

			/* R ← (1−α)·R + α·x·P  (Eq. 5, second line) */
			rr = a1 * rr + a * x * pr;
			ri = a1 * ri + a * x * pi;

			/* R̃ ← (1−α)·R̃ + α·R  (Eq. 8, optional second smoothing) */
			rtr = a1 * rtr + a * rr;
			rti = a1 * rti + a * ri;
		}

		/* renormalise phasor to prevent slow magnitude drift from
		   accumulated floating-point rounding in repeated complex mults */
		float pnorm = 1.0f / sqrtf(pr*pr + pi*pi);
		ctx->res_p_re[k]  = pr  * pnorm;
		ctx->res_p_im[k]  = pi  * pnorm;
		ctx->res_r_re[k]  = rr;
		ctx->res_r_im[k]  = ri;
		ctx->res_rt_re[k] = rtr;
		ctx->res_rt_im[k] = rti;
	}

	/* map resonators → display rows ─────────────────────────────────── */
	uint32_t N    = ctx->resonate_bins;
	float    f0   = ctx->res_freq[0];
	float    f1   = ctx->res_freq[N - 1];
	float    lf0  = logf(f0 > 0.f ? f0 : 1e-6f);
	float    lf1  = logf(f1 > 0.f ? f1 : 1e-6f);
	float    span = lf1 - lf0;

	float    disp_f0   = ctx->min_freq < 1.0f ? 1.0f : ctx->min_freq;
	float    disp_f1   = ctx->max_freq;
	float    log_lo    = logf(disp_f0);
	float    log_hi    = logf(disp_f1);

	for (uint32_t y = 0; y < ctx->height; y++) {
		/* y = 0 → top → high frequency */
		float t = 1.0f - (float)y / (float)(ctx->height - 1);
		float log_freq = ctx->log_scale
			? log_lo + t * (log_hi - log_lo)
			: logf(disp_f0 + t * (disp_f1 - disp_f0));

		/* fractional resonator index (resonators are log-spaced) */
		float kf = (span > 0.f)
			? (log_freq - lf0) / span * (float)(N - 1)
			: 0.0f;
		kf = clamp_index(kf, N);
		int   k0   = (int)kf;
		float frac = kf - (float)k0;
		int   k1   = k0 + 1 < (int)N ? k0 + 1 : k0;

		float re0, im0, re1, im1;
		if (ctx->resonate_smooth) {
			re0 = ctx->res_rt_re[k0]; im0 = ctx->res_rt_im[k0];
			re1 = ctx->res_rt_re[k1]; im1 = ctx->res_rt_im[k1];
		} else {
			re0 = ctx->res_r_re[k0];  im0 = ctx->res_r_im[k0];
			re1 = ctx->res_r_re[k1];  im1 = ctx->res_r_im[k1];
		}

		float mag0 = sqrtf(re0*re0 + im0*im0);
		float mag1 = sqrtf(re1*re1 + im1*im1);
		float mag  = mag0 + frac * (mag1 - mag0);

		float db = (mag > 1e-12f)
			? 20.0f * log10f(mag) + ctx->gain_db
			: ctx->db_min;

		ctx->col_pixels[y] = palette_lookup(db, ctx->db_min, ctx->db_max,
		                                    ctx->palette, ctx->palette_count);
	}

	return true;
}

static bool compute_column(struct spectrogram_source *ctx)
{
	return ctx->analysis_mode == ANALYSIS_RESONATE
		? compute_column_resonate(ctx)
		: compute_column_fft(ctx);
}

/* ── texture update (graphics thread) ────────────────────────────────────── */

static void write_column(struct spectrogram_source *ctx)
{
	uint32_t xc = ctx->write_col;
	for (uint32_t y = 0; y < ctx->height; y++)
		ctx->tex_cpu[y * ctx->width + xc] = ctx->col_pixels[y];

	ctx->write_col = (xc + 1) % ctx->width;
}

static void upload_texture(struct spectrogram_source *ctx)
{
	uint8_t  *ptr;
	uint32_t  ls;

	obs_enter_graphics();
	if (gs_texture_map(ctx->tex, &ptr, &ls)) {
		for (uint32_t y = 0; y < ctx->height; y++)
			memcpy(ptr + y * ls,
			       ctx->tex_cpu + y * ctx->width,
			       ctx->width * 4);
		gs_texture_unmap(ctx->tex);
	}
	obs_leave_graphics();
}

static void reset_ring(struct spectrogram_source *ctx)
{
	pthread_mutex_lock(&ctx->mutex);
	ctx->ring_r = ctx->ring_w = ctx->ring_n = 0;
	pthread_mutex_unlock(&ctx->mutex);
}

/* ── audio source connection ─────────────────────────────────────────────── */

static bool is_output_mix(const char *ref)
{
	return ref && strcmp(ref, OUTPUT_MIX_ID) == 0;
}

/* Resolve the setting value to a source.  New settings store the UUID;
   settings saved by older versions store the source name. */
static obs_source_t *find_audio_source(const char *ref)
{
	obs_source_t *src = obs_get_source_by_uuid(ref);
	return src ? src : obs_get_source_by_name(ref);
}

static bool is_audio_connected(const struct spectrogram_source *ctx)
{
	return ctx->audio_weak || ctx->output_mix_connected;
}

static void handle_audio_source_removed(void *param, calldata_t *cd)
{
	(void)cd;
	struct spectrogram_source *ctx = param;
	os_atomic_set_bool(&ctx->audio_removed, true);
}

static void disconnect_audio(struct spectrogram_source *ctx)
{
	if (ctx->output_mix_connected) {
		audio_output_disconnect(obs_get_audio(), OUTPUT_MIX_TRACK,
		                        capture_output_mix, ctx);
		ctx->output_mix_connected = false;
	}
	if (ctx->audio_weak) {
		obs_source_t *src = obs_weak_source_get_source(ctx->audio_weak);
		if (src) {
			obs_source_remove_audio_capture_callback(src,
				capture_source_audio, ctx);
			signal_handler_disconnect(obs_source_get_signal_handler(src),
				"remove", handle_audio_source_removed, ctx);
			obs_source_release(src);
		}
		obs_weak_source_release(ctx->audio_weak);
		ctx->audio_weak = NULL;
		os_atomic_set_bool(&ctx->audio_removed, false);
	}
}

/* Rewrite a legacy name-based setting to the source's UUID, so renaming
   the audio source later doesn't break the link. */
static void migrate_audio_ref(struct spectrogram_source *ctx, obs_source_t *src)
{
	const char *uuid = obs_source_get_uuid(src);
	if (!uuid || strcmp(uuid, ctx->audio_ref) == 0)
		return;

	obs_data_t *settings = obs_source_get_settings(ctx->source);
	obs_data_set_string(settings, S_AUDIO_SRC, uuid);
	obs_data_release(settings);

	bfree(ctx->audio_ref);
	ctx->audio_ref = bstrdup(uuid);
}

/* Try to connect to ctx->audio_ref.  Failing is normal: while a scene
   collection loads, the audio source may not exist yet, so the tick
   retries every RECONNECT_INTERVAL. */
static void connect_audio(struct spectrogram_source *ctx)
{
	if (!ctx->audio_ref || !*ctx->audio_ref || is_audio_connected(ctx))
		return;

	if (is_output_mix(ctx->audio_ref)) {
		ctx->output_mix_connected = audio_output_connect(obs_get_audio(),
			OUTPUT_MIX_TRACK, NULL, capture_output_mix, ctx);
		return;
	}

	obs_source_t *src = find_audio_source(ctx->audio_ref);
	if (!src)
		return;

	if (src != ctx->source && !obs_source_removed(src)) {
		migrate_audio_ref(ctx, src);
		obs_source_add_audio_capture_callback(src, capture_source_audio, ctx);
		signal_handler_connect(obs_source_get_signal_handler(src),
			"remove", handle_audio_source_removed, ctx);
		ctx->audio_weak = obs_source_get_weak_source(src);
	}
	obs_source_release(src);
}

/* Drop the connection if the audio source was deleted, and keep retrying
   while the configured source is missing.  This runs every frame, so it
   must not take a strong reference: holding one, even briefly, can stop
   the audio source from being freed when OBS clears a scene collection. */
static void check_audio_connection(struct spectrogram_source *ctx, float seconds)
{
	if (ctx->audio_weak && (os_atomic_load_bool(&ctx->audio_removed) ||
	                        obs_weak_source_expired(ctx->audio_weak)))
		disconnect_audio(ctx);

	if (is_audio_connected(ctx))
		return;

	ctx->reconnect_timer -= seconds;
	if (ctx->reconnect_timer <= 0.0f) {
		ctx->reconnect_timer = RECONNECT_INTERVAL;
		connect_audio(ctx);
	}
}

/* ── tick (graphics thread) ──────────────────────────────────────────────── *
 * Analysis runs here rather than in video_render: render is called once per *
 * view (preview, program, projectors), tick exactly once per frame.         */

static void spectrogram_tick(void *data, float seconds)
{
	struct spectrogram_source *ctx = data;

	check_audio_connection(ctx, seconds);

	if (!ctx->tex || !ctx->tex_cpu)
		return;

	/* nothing sees us: don't burn CPU, just discard the audio */
	if (!obs_source_showing(ctx->source)) {
		reset_ring(ctx);
		return;
	}

	bool dirty = false;
	for (uint32_t i = 0; i < ctx->width && compute_column(ctx); i++) {
		write_column(ctx);
		dirty = true;
	}
	if (dirty)
		upload_texture(ctx);
}

/* ── rendering (graphics thread) ─────────────────────────────────────────── */

static void draw_quad(float sx, float sy, float sw, float sh,
                      float u0, float v0, float u1, float v1)
{
	gs_render_start(false);

	gs_texcoord(u0, v0, 0); gs_vertex2f(sx,      sy);
	gs_texcoord(u1, v0, 0); gs_vertex2f(sx + sw, sy);
	gs_texcoord(u0, v1, 0); gs_vertex2f(sx,      sy + sh);

	gs_texcoord(u1, v0, 0); gs_vertex2f(sx + sw, sy);
	gs_texcoord(u1, v1, 0); gs_vertex2f(sx + sw, sy + sh);
	gs_texcoord(u0, v1, 0); gs_vertex2f(sx,      sy + sh);

	gs_render_stop(GS_TRIS);
}

static void spectrogram_render(void *data, gs_effect_t *unused_effect)
{
	(void)unused_effect;
	struct spectrogram_source *ctx = data;

	if (!ctx->tex || !ctx->effect)
		return;

	/* U starts at the write head (oldest column) and runs to write_head + 1.0.
	   AddressU = Wrap in the shader makes the sampler roll over seamlessly,
	   so screen pixel p maps to texture column (write_col + p) % width. */
	float u0 = (float)ctx->write_col / (float)ctx->width;

	gs_effect_set_texture(ctx->ep_image, ctx->tex);

	while (gs_effect_loop(ctx->effect, "Draw")) {
		draw_quad(0.f, 0.f, (float)ctx->width, (float)ctx->height,
		          u0, 0.f, u0 + 1.0f, 1.f);
	}
}

/* ── texture (re)creation ────────────────────────────────────────────────── */

static void rebuild_texture(struct spectrogram_source *ctx)
{
	obs_enter_graphics();
	if (ctx->tex) {
		gs_texture_destroy(ctx->tex);
		ctx->tex = NULL;
	}
	bfree(ctx->tex_cpu);
	ctx->tex_cpu = bzalloc(ctx->width * ctx->height * sizeof(uint32_t));
	ctx->tex     = gs_texture_create(ctx->width, ctx->height,
	                                  GS_BGRA, 1, NULL, GS_DYNAMIC);
	ctx->write_col = 0;
	obs_leave_graphics();
}

/* ── properties ──────────────────────────────────────────────────────────── */

struct audio_list_ctx {
	obs_property_t *list;
	obs_source_t   *self;
};

/* Offer the same sources the Audio Mixer would show.  OBS_SOURCE_AUDIO
   is a static capability flag, so it's also set on webcams with audio
   disabled, window/game captures not capturing audio, browser sources not
   routing audio through OBS, etc.  Those clear obs_source_audio_active(). */
static bool add_audio_source_to_list(void *param, obs_source_t *src)
{
	struct audio_list_ctx *d = param;

	if (src == d->self)
		return true;
	if (!(obs_source_get_output_flags(src) & OBS_SOURCE_AUDIO))
		return true;
	if (!obs_source_audio_active(src))
		return true;

	obs_property_list_add_string(d->list, obs_source_get_name(src),
	                             obs_source_get_uuid(src));
	return true;
}

/* Keep the current selection in the list even if it's filtered out or
   missing, so opening the properties doesn't silently change it. */
static void add_current_audio_ref(obs_property_t *list, const char *ref)
{
	if (!ref || !*ref || is_output_mix(ref))
		return;

	size_t n = obs_property_list_item_count(list);
	for (size_t i = 0; i < n; i++) {
		if (strcmp(obs_property_list_item_string(list, i), ref) == 0)
			return;
	}

	obs_source_t *src = find_audio_source(ref);
	struct dstr label = {0};
	if (src) {
		dstr_printf(&label, "%s %s", obs_source_get_name(src),
		            obs_module_text("AudioSource.Inactive"));
		obs_source_release(src);
	} else {
		dstr_printf(&label, "%s", obs_module_text("AudioSource.Missing"));
	}
	obs_property_list_add_string(list, label.array, ref);
	dstr_free(&label);
}

static bool analysis_mode_changed(obs_properties_t *props,
                                  obs_property_t   *p,
                                  obs_data_t       *settings)
{
	(void)p;
	int  mode        = (int)obs_data_get_int(settings, S_ANALYSIS_MODE);
	bool is_fft      = (mode == ANALYSIS_FFT);
	bool is_resonate = (mode == ANALYSIS_RESONATE);

	obs_property_set_visible(obs_properties_get(props, S_FFT_SIZE),       is_fft);
	obs_property_set_visible(obs_properties_get(props, S_OVERLAP),         is_fft);
	obs_property_set_visible(obs_properties_get(props, S_WINDOW),          is_fft);
	obs_property_set_visible(obs_properties_get(props, S_RESONATE_BINS),   is_resonate);
	obs_property_set_visible(obs_properties_get(props, S_RESONATE_HOP),    is_resonate);
	obs_property_set_visible(obs_properties_get(props, S_RESONATE_SMOOTH), is_resonate);
	return true;
}

static obs_properties_t *spectrogram_get_properties(void *data)
{
	struct spectrogram_source *ctx = data;
	obs_properties_t *props = obs_properties_create();

	/* audio source */
	obs_property_t *src_p = obs_properties_add_list(props, S_AUDIO_SRC,
		T("AudioSource"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_set_long_description(src_p, T("AudioSource.Description"));
	obs_property_list_add_string(src_p, T("AudioSource.None"), "");
	obs_property_list_add_string(src_p, T("AudioSource.OutputMix"),
	                             OUTPUT_MIX_ID);
	struct audio_list_ctx list_ctx = {
		.list = src_p,
		.self = ctx ? ctx->source : NULL,
	};
	obs_enum_sources(add_audio_source_to_list, &list_ctx);
	if (ctx)
		add_current_audio_ref(src_p, ctx->audio_ref);

	/* display */
	obs_properties_add_int(props, S_WIDTH,  T("Width"),  64, 4096, 1);
	obs_properties_add_int(props, S_HEIGHT, T("Height"), 64, MAX_HEIGHT, 1);

	/* analysis mode */
	obs_property_t *mode_p = obs_properties_add_list(props, S_ANALYSIS_MODE,
		T("AnalysisMode"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(mode_p, T("AnalysisMode.FFT"),      ANALYSIS_FFT);
	obs_property_list_add_int(mode_p, T("AnalysisMode.Resonate"), ANALYSIS_RESONATE);
	obs_property_set_modified_callback(mode_p, analysis_mode_changed);

	/* ── FFT options (hidden when Resonate is active) ─────────────────── */
	obs_property_t *fft_p = obs_properties_add_list(props, S_FFT_SIZE,
		T("FFTSize"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(fft_p, "256",  256);
	obs_property_list_add_int(fft_p, "512",  512);
	obs_property_list_add_int(fft_p, "1024", 1024);
	obs_property_list_add_int(fft_p, "2048", 2048);
	obs_property_list_add_int(fft_p, "4096", 4096);
	obs_property_list_add_int(fft_p, "8192", 8192);

	obs_properties_add_float_slider(props, S_OVERLAP,
		T("Overlap"), 0.0, 0.95, 0.05);

	obs_property_t *win_p = obs_properties_add_list(props, S_WINDOW,
		T("Window"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(win_p, T("Window.Rectangular"), WIN_RECT);
	obs_property_list_add_int(win_p, T("Window.Hann"),        WIN_HANN);
	obs_property_list_add_int(win_p, T("Window.Hamming"),     WIN_HAMMING);
	obs_property_list_add_int(win_p, T("Window.Blackman"),    WIN_BLACKMAN);

	/* ── Resonate options (hidden when FFT is active) ─────────────────── */
	obs_property_t *bins_p = obs_properties_add_int_slider(props,
		S_RESONATE_BINS, T("Resonators"), 2, MAX_RESONATE_BINS, 1);
	obs_property_set_visible(bins_p, false);

	obs_property_t *hop_p = obs_properties_add_list(props, S_RESONATE_HOP,
		T("ColumnHop"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(hop_p, "64",   64);
	obs_property_list_add_int(hop_p, "128",  128);
	obs_property_list_add_int(hop_p, "256",  256);
	obs_property_list_add_int(hop_p, "512",  512);
	obs_property_list_add_int(hop_p, "1024", 1024);
	obs_property_list_add_int(hop_p, "2048", 2048);
	obs_property_set_visible(hop_p, false);

	obs_property_t *smooth_p = obs_properties_add_bool(props,
		S_RESONATE_SMOOTH, T("DoubleSmoothing"));
	obs_property_set_visible(smooth_p, false);

	/* ── common ───────────────────────────────────────────────────────── */
	obs_properties_add_float(props, S_MIN_FREQ, T("MinFrequency"),
	                          1.0, 20000.0, 1.0);
	obs_properties_add_float(props, S_MAX_FREQ, T("MaxFrequency"),
	                          100.0, 24000.0, 1.0);
	obs_properties_add_float_slider(props, S_DB_MIN, T("FloorDb"),
	                                 -120.0, 0.0, 1.0);
	obs_properties_add_float_slider(props, S_DB_MAX, T("CeilingDb"),
	                                 -120.0, 0.0, 1.0);
	obs_properties_add_float_slider(props, S_GAIN, T("GainDb"),
	                                 -40.0, 40.0, 0.5);
	obs_properties_add_bool(props, S_LOG_SCALE, T("LogScale"));

	/* palette – 3 fixed color stops: low power → mid → high power */
	obs_properties_add_color_alpha(props, S_COLOR_0, T("ColorLow"));
	obs_properties_add_color_alpha(props, S_COLOR_1, T("ColorMid"));
	obs_properties_add_color_alpha(props, S_COLOR_2, T("ColorHigh"));

	return props;
}

static void spectrogram_get_defaults(obs_data_t *settings)
{
	obs_data_set_default_string(settings, S_AUDIO_SRC,      "");
	obs_data_set_default_int   (settings, S_WIDTH,          800);
	obs_data_set_default_int   (settings, S_HEIGHT,         300);
	obs_data_set_default_int   (settings, S_ANALYSIS_MODE,  ANALYSIS_FFT);
	/* FFT */
	obs_data_set_default_int   (settings, S_FFT_SIZE,       2048);
	obs_data_set_default_double(settings, S_OVERLAP,        0.75);
	obs_data_set_default_int   (settings, S_WINDOW,         WIN_HANN);
	/* Resonate */
	obs_data_set_default_int   (settings, S_RESONATE_BINS,  120);
	obs_data_set_default_int   (settings, S_RESONATE_HOP,   512);
	obs_data_set_default_bool  (settings, S_RESONATE_SMOOTH, false);
	/* common */
	obs_data_set_default_double(settings, S_MIN_FREQ,       20.0);
	obs_data_set_default_double(settings, S_MAX_FREQ,       20000.0);
	obs_data_set_default_double(settings, S_DB_MIN,        -90.0);
	obs_data_set_default_double(settings, S_DB_MAX,          0.0);
	obs_data_set_default_double(settings, S_GAIN,            0.0);
	obs_data_set_default_bool  (settings, S_LOG_SCALE,       true);

	/* default palette: black → yellow → white.
	   OBS stores colors as R|(G<<8)|(B<<16)|(A<<24) (ABGR). */
	obs_data_set_default_int(settings, S_COLOR_0, 0xFF000000); /* black  */
	obs_data_set_default_int(settings, S_COLOR_1, 0xFF00FFFF); /* yellow */
	obs_data_set_default_int(settings, S_COLOR_2, 0xFFFFFFFF); /* white  */
}

/* ── update ──────────────────────────────────────────────────────────────── */

static void spectrogram_update(void *data, obs_data_t *settings)
{
	struct spectrogram_source *ctx = data;

	uint32_t new_w = (uint32_t)obs_data_get_int(settings, S_WIDTH);
	uint32_t new_h = (uint32_t)obs_data_get_int(settings, S_HEIGHT);
	if (!new_w) new_w = 800;
	if (!new_h) new_h = 300;
	if (new_h > MAX_HEIGHT) new_h = MAX_HEIGHT;

	bool rebuild = (new_w != ctx->width || new_h != ctx->height || !ctx->tex);
	ctx->width  = new_w;
	ctx->height = new_h;

	ctx->analysis_mode = (int)obs_data_get_int(settings, S_ANALYSIS_MODE);

	/* FFT settings */
	uint32_t new_fft = (uint32_t)obs_data_get_int(settings, S_FFT_SIZE);
	if (!new_fft || new_fft > MAX_FFT_SIZE) new_fft = 2048;
	ctx->fft_size = new_fft;

	float overlap = (float)obs_data_get_double(settings, S_OVERLAP);
	ctx->hop_size = (uint32_t)((1.0f - overlap) * (float)ctx->fft_size);
	if (ctx->hop_size < 1) ctx->hop_size = 1;

	ctx->window_type = (int)obs_data_get_int(settings, S_WINDOW);
	build_window(ctx->win, ctx->fft_size, ctx->window_type);

	/* Resonate settings */
	uint32_t rbins = (uint32_t)obs_data_get_int(settings, S_RESONATE_BINS);
	if (!rbins || rbins > MAX_RESONATE_BINS) rbins = 120;
	ctx->resonate_bins   = rbins;

	uint32_t rhop = (uint32_t)obs_data_get_int(settings, S_RESONATE_HOP);
	if (!rhop || rhop > MAX_FFT_SIZE) rhop = 512;
	ctx->resonate_hop    = rhop;

	ctx->resonate_smooth = obs_data_get_bool(settings, S_RESONATE_SMOOTH);

	/* common */
	ctx->log_scale = obs_data_get_bool  (settings, S_LOG_SCALE);
	ctx->min_freq  = (float)obs_data_get_double(settings, S_MIN_FREQ);
	ctx->max_freq  = (float)obs_data_get_double(settings, S_MAX_FREQ);

	/* the two fields are independent in the UI, so the range can arrive
	   inverted or reaching past Nyquist */
	float nyquist = (float)ctx->sample_rate * 0.5f;
	if (ctx->min_freq > ctx->max_freq) {
		float t = ctx->min_freq;
		ctx->min_freq = ctx->max_freq;
		ctx->max_freq = t;
	}
	if (ctx->max_freq > nyquist)       ctx->max_freq = nyquist;
	if (ctx->min_freq < 1.0f)          ctx->min_freq = 1.0f;
	if (ctx->min_freq >= ctx->max_freq) ctx->min_freq = ctx->max_freq * 0.5f;

	ctx->db_min    = (float)obs_data_get_double(settings, S_DB_MIN);
	ctx->db_max    = (float)obs_data_get_double(settings, S_DB_MAX);
	ctx->gain_db   = (float)obs_data_get_double(settings, S_GAIN);

	/* read 3 fixed colour stops and convert OBS ABGR → GS_BGRA uint32.
	   OBS stores colours as R|(G<<8)|(B<<16)|(A<<24); GS_BGRA uint32 is
	   B|(G<<8)|(R<<16)|(A<<24), so R and B bytes need swapping. */
	ctx->palette_count = 3;
	{
		static const char *keys[3] = {S_COLOR_0, S_COLOR_1, S_COLOR_2};
		for (int i = 0; i < 3; i++) {
			uint32_t abgr = (uint32_t)obs_data_get_int(settings, keys[i]);
			uint8_t r =  abgr        & 0xFF;
			uint8_t g = (abgr >>  8) & 0xFF;
			uint8_t b = (abgr >> 16) & 0xFF;
			uint8_t a = (abgr >> 24) & 0xFF;
			ctx->palette[i] = b | ((uint32_t)g << 8)
			                    | ((uint32_t)r << 16)
			                    | ((uint32_t)a << 24);
		}
	}

	/* audio source */
	const char *ref = obs_data_get_string(settings, S_AUDIO_SRC);
	if (!ctx->audio_ref || strcmp(ref, ctx->audio_ref) != 0) {
		disconnect_audio(ctx);
		bfree(ctx->audio_ref);
		ctx->audio_ref = bstrdup(ref);
		connect_audio(ctx);
		ctx->reconnect_timer = RECONNECT_INTERVAL;
	}

	if (ctx->analysis_mode == ANALYSIS_RESONATE)
		init_resonators(ctx);

	if (rebuild)
		rebuild_texture(ctx);

	reset_ring(ctx);
}

/* ── lifecycle ───────────────────────────────────────────────────────────── */

static void *spectrogram_create(obs_data_t *settings, obs_source_t *source)
{
	struct spectrogram_source *ctx = bzalloc(sizeof(*ctx));
	ctx->source = source;
	pthread_mutex_init(&ctx->mutex, NULL);

	struct obs_audio_info ai;
	if (obs_get_audio_info(&ai)) {
		ctx->sample_rate = ai.samples_per_sec;
		ctx->channels    = get_audio_channels(ai.speakers);
	} else {
		ctx->sample_rate = 48000;
		ctx->channels    = 2;
	}
	if (ctx->channels > MAX_AV_PLANES)
		ctx->channels = MAX_AV_PLANES;

	obs_enter_graphics();
	char *path = obs_module_file("shaders/spectrogram.effect");
	ctx->effect   = gs_effect_create_from_file(path, NULL);
	bfree(path);
	if (ctx->effect)
		ctx->ep_image = gs_effect_get_param_by_name(ctx->effect, "image");
	obs_leave_graphics();

	spectrogram_update(ctx, settings);
	return ctx;
}

static void spectrogram_destroy(void *data)
{
	struct spectrogram_source *ctx = data;
	disconnect_audio(ctx);
	bfree(ctx->audio_ref);

	obs_enter_graphics();
	if (ctx->tex)    gs_texture_destroy(ctx->tex);
	if (ctx->effect) gs_effect_destroy(ctx->effect);
	obs_leave_graphics();

	bfree(ctx->tex_cpu);
	pthread_mutex_destroy(&ctx->mutex);
	bfree(ctx);
}

static uint32_t spectrogram_get_width (void *data)
{
	return ((struct spectrogram_source *)data)->width;
}
static uint32_t spectrogram_get_height(void *data)
{
	return ((struct spectrogram_source *)data)->height;
}
static const char *spectrogram_get_name(void *unused)
{
	(void)unused;
	return T("Spectrogram");
}

/* ── registration ────────────────────────────────────────────────────────── */

struct obs_source_info spectrogram_source_info = {
	.id             = "spectrogram_source",
	.type           = OBS_SOURCE_TYPE_INPUT,
	.output_flags   = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW,
	.icon_type      = OBS_ICON_TYPE_AUDIO_OUTPUT,
	.get_name       = spectrogram_get_name,
	.create         = spectrogram_create,
	.destroy        = spectrogram_destroy,
	.update         = spectrogram_update,
	.video_tick     = spectrogram_tick,
	.video_render   = spectrogram_render,
	.get_properties = spectrogram_get_properties,
	.get_defaults   = spectrogram_get_defaults,
	.get_width      = spectrogram_get_width,
	.get_height     = spectrogram_get_height,
};
