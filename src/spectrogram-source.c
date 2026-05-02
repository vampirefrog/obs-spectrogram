#include "spectrogram-source.h"
#include "fft.h"

#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/bmem.h>
#include <util/platform.h>

#include <util/threading.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── limits ─────────────────────────────────────────────────────────────── */

#define MAX_FFT_SIZE      8192
#define MAX_HEIGHT        8192
#define MAX_PALETTE       16
#define RING_SIZE         (MAX_FFT_SIZE * 8)
#define MAX_RESONATE_BINS 1024

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
	obs_source_t *audio_source;
	char         *audio_source_name;

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

	/* pending column – produced then consumed on render thread */
	uint32_t col_pixels[MAX_HEIGHT];
	bool     col_ready;

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

static void audio_capture_cb(void *param, obs_source_t *source,
                              const struct audio_data *data, bool muted)
{
	(void)source;
	struct spectrogram_source *ctx = param;
	if (muted || !data->data[0])
		return;

	const float *in = (const float *)data->data[0];
	uint32_t     nf = data->frames;

	pthread_mutex_lock(&ctx->mutex);
	for (uint32_t i = 0; i < nf; i++) {
		ctx->ring[ctx->ring_w] = in[i];
		ctx->ring_w = (ctx->ring_w + 1) % RING_SIZE;
		if (ctx->ring_n < RING_SIZE) {
			ctx->ring_n++;
		} else {
			ctx->ring_r = (ctx->ring_r + 1) % RING_SIZE;
		}
	}
	pthread_mutex_unlock(&ctx->mutex);
}

/* ── FFT column (render thread) ──────────────────────────────────────────── */

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

		float bin_f = freq * (float)ctx->fft_size / (float)ctx->sample_rate;
		int   b0    = (int)bin_f;
		float frac  = bin_f - (float)b0;
		int   b1    = b0 + 1;

		if (b0 < 0)              b0 = 0;
		if (b0 >= (int)num_bins) b0 = (int)num_bins - 1;
		if (b1 >= (int)num_bins) b1 = (int)num_bins - 1;

		float db = ctx->fft_im[b0] + frac * (ctx->fft_im[b1] - ctx->fft_im[b0]);
		ctx->col_pixels[y] = palette_lookup(db, ctx->db_min, ctx->db_max,
		                                    ctx->palette, ctx->palette_count);
	}

	ctx->col_ready = true;
	return true;
}

/* ── Resonate column (render thread) ─────────────────────────────────────── *
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
		float kf   = (span > 0.f)
			? (log_freq - lf0) / span * (float)(N - 1)
			: 0.0f;
		int   k0   = (int)kf;
		float frac = kf - (float)k0;
		int   k1   = k0 + 1;

		if (k0 < 0)       { k0 = 0;     frac = 0.f; }
		if (k0 >= (int)N) { k0 = (int)N - 1; frac = 0.f; }
		if (k1 >= (int)N)   k1 = (int)N - 1;

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

	ctx->col_ready = true;
	return true;
}

static bool compute_column(struct spectrogram_source *ctx)
{
	return ctx->analysis_mode == ANALYSIS_RESONATE
		? compute_column_resonate(ctx)
		: compute_column_fft(ctx);
}

/* ── texture update (render thread, inside graphics context) ────────────── */

static void flush_column(struct spectrogram_source *ctx)
{
	if (!ctx->col_ready || !ctx->tex || !ctx->tex_cpu)
		return;

	uint32_t xc = ctx->write_col;
	for (uint32_t y = 0; y < ctx->height; y++)
		ctx->tex_cpu[y * ctx->width + xc] = ctx->col_pixels[y];

	ctx->write_col = (xc + 1) % ctx->width;
	ctx->col_ready = false;

	uint8_t  *ptr;
	uint32_t  ls;
	if (gs_texture_map(ctx->tex, &ptr, &ls)) {
		for (uint32_t y = 0; y < ctx->height; y++)
			memcpy(ptr + y * ls,
			       ctx->tex_cpu + y * ctx->width,
			       ctx->width * 4);
		gs_texture_unmap(ctx->tex);
	}
}

/* ── rendering (render thread) ───────────────────────────────────────────── */

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

	for (int i = 0; i < 8; i++) {
		if (!compute_column(ctx))
			break;
		flush_column(ctx);
	}

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

/* ── audio source attachment ─────────────────────────────────────────────── */

static void detach_audio(struct spectrogram_source *ctx)
{
	if (ctx->audio_source) {
		obs_source_remove_audio_capture_callback(ctx->audio_source,
		                                          audio_capture_cb, ctx);
		obs_source_release(ctx->audio_source);
		ctx->audio_source = NULL;
	}
	bfree(ctx->audio_source_name);
	ctx->audio_source_name = NULL;
}

static void attach_audio(struct spectrogram_source *ctx, const char *name)
{
	detach_audio(ctx);
	if (!name || !*name)
		return;

	ctx->audio_source_name = bstrdup(name);
	ctx->audio_source      = obs_get_source_by_name(name);
	if (!ctx->audio_source)
		return;

	obs_source_add_audio_capture_callback(ctx->audio_source,
	                                       audio_capture_cb, ctx);

	struct obs_audio_info ai;
	ctx->sample_rate = obs_get_audio_info(&ai) ? ai.samples_per_sec : 48000;
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

static bool enum_audio_cb(void *param, obs_source_t *src)
{
	if (obs_source_get_output_flags(src) & OBS_SOURCE_AUDIO) {
		const char *name = obs_source_get_name(src);
		obs_property_list_add_string((obs_property_t *)param, name, name);
	}
	return true;
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
		"Audio Source", OBS_COMBO_TYPE_EDITABLE, OBS_COMBO_FORMAT_STRING);
	obs_enum_sources(enum_audio_cb, src_p);

	/* display */
	obs_properties_add_int(props, S_WIDTH,  "Width (px)",  64, 4096, 1);
	obs_properties_add_int(props, S_HEIGHT, "Height (px)", 64, MAX_HEIGHT, 1);

	/* analysis mode */
	obs_property_t *mode_p = obs_properties_add_list(props, S_ANALYSIS_MODE,
		"Analysis Mode", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(mode_p, "FFT",     ANALYSIS_FFT);
	obs_property_list_add_int(mode_p, "Resonate (François 2025)", ANALYSIS_RESONATE);
	obs_property_set_modified_callback(mode_p, analysis_mode_changed);

	/* ── FFT options (hidden when Resonate is active) ─────────────────── */
	obs_property_t *fft_p = obs_properties_add_list(props, S_FFT_SIZE,
		"FFT Size", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(fft_p, "256",  256);
	obs_property_list_add_int(fft_p, "512",  512);
	obs_property_list_add_int(fft_p, "1024", 1024);
	obs_property_list_add_int(fft_p, "2048", 2048);
	obs_property_list_add_int(fft_p, "4096", 4096);
	obs_property_list_add_int(fft_p, "8192", 8192);

	obs_properties_add_float_slider(props, S_OVERLAP,
		"Overlap", 0.0, 0.95, 0.05);

	obs_property_t *win_p = obs_properties_add_list(props, S_WINDOW,
		"Window Function", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(win_p, "Rectangular", WIN_RECT);
	obs_property_list_add_int(win_p, "Hann",        WIN_HANN);
	obs_property_list_add_int(win_p, "Hamming",     WIN_HAMMING);
	obs_property_list_add_int(win_p, "Blackman",    WIN_BLACKMAN);

	/* ── Resonate options (hidden when FFT is active) ─────────────────── */
	obs_property_t *bins_p = obs_properties_add_int_slider(props,
		S_RESONATE_BINS, "Resonators", 2, MAX_RESONATE_BINS, 1);
	obs_property_set_visible(bins_p, false);

	obs_property_t *hop_p = obs_properties_add_list(props, S_RESONATE_HOP,
		"Column Hop (samples)", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(hop_p, "64",   64);
	obs_property_list_add_int(hop_p, "128",  128);
	obs_property_list_add_int(hop_p, "256",  256);
	obs_property_list_add_int(hop_p, "512",  512);
	obs_property_list_add_int(hop_p, "1024", 1024);
	obs_property_list_add_int(hop_p, "2048", 2048);
	obs_property_set_visible(hop_p, false);

	obs_property_t *smooth_p = obs_properties_add_bool(props,
		S_RESONATE_SMOOTH, "Double Smoothing (R̃, Eq. 8)");
	obs_property_set_visible(smooth_p, false);

	/* ── common ───────────────────────────────────────────────────────── */
	obs_properties_add_float(props, S_MIN_FREQ, "Min Frequency (Hz)",
	                          1.0, 20000.0, 1.0);
	obs_properties_add_float(props, S_MAX_FREQ, "Max Frequency (Hz)",
	                          100.0, 24000.0, 1.0);
	obs_properties_add_float_slider(props, S_DB_MIN, "Floor (dB)",
	                                 -120.0, 0.0, 1.0);
	obs_properties_add_float_slider(props, S_DB_MAX, "Ceiling (dB)",
	                                 -120.0, 0.0, 1.0);
	obs_properties_add_float_slider(props, S_GAIN, "Gain (dB)",
	                                 -40.0, 40.0, 0.5);
	obs_properties_add_bool(props, S_LOG_SCALE, "Logarithmic Frequency Scale");

	/* palette – 3 fixed color stops: low power → mid → high power */
	obs_properties_add_color_alpha(props, S_COLOR_0, "Color: Low Power");
	obs_properties_add_color_alpha(props, S_COLOR_1, "Color: Mid Power");
	obs_properties_add_color_alpha(props, S_COLOR_2, "Color: High Power");

	return props;
}

static void spectrogram_get_defaults(obs_data_t *settings)
{
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

	/* audio source (sets sample_rate) */
	const char *src_name = obs_data_get_string(settings, S_AUDIO_SRC);
	if (!ctx->audio_source_name ||
	    strcmp(src_name, ctx->audio_source_name) != 0)
		attach_audio(ctx, src_name);

	/* initialise resonators after sample_rate is known */
	if (ctx->analysis_mode == ANALYSIS_RESONATE)
		init_resonators(ctx);

	if (rebuild)
		rebuild_texture(ctx);

	pthread_mutex_lock(&ctx->mutex);
	ctx->ring_r = ctx->ring_w = ctx->ring_n = 0;
	pthread_mutex_unlock(&ctx->mutex);
}

/* ── lifecycle ───────────────────────────────────────────────────────────── */

static void *spectrogram_create(obs_data_t *settings, obs_source_t *source)
{
	struct spectrogram_source *ctx = bzalloc(sizeof(*ctx));
	ctx->source = source;
	pthread_mutex_init(&ctx->mutex, NULL);

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
	detach_audio(ctx);

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
	return "Spectrogram";
}

/* ── registration ────────────────────────────────────────────────────────── */

struct obs_source_info spectrogram_source_info = {
	.id             = "spectrogram_source",
	.type           = OBS_SOURCE_TYPE_INPUT,
	.output_flags   = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW,
	.get_name       = spectrogram_get_name,
	.create         = spectrogram_create,
	.destroy        = spectrogram_destroy,
	.update         = spectrogram_update,
	.video_render   = spectrogram_render,
	.get_properties = spectrogram_get_properties,
	.get_defaults   = spectrogram_get_defaults,
	.get_width      = spectrogram_get_width,
	.get_height     = spectrogram_get_height,
};
