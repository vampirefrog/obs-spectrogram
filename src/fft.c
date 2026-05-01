#include "fft.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static uint32_t bit_reverse(uint32_t x, uint32_t bits)
{
	uint32_t result = 0;
	for (uint32_t i = 0; i < bits; i++) {
		result = (result << 1) | (x & 1);
		x >>= 1;
	}
	return result;
}

void fft_forward(float *re, float *im, uint32_t n)
{
	uint32_t bits = 0;
	for (uint32_t t = n; t > 1; t >>= 1)
		bits++;

	/* Bit-reversal permutation */
	for (uint32_t i = 0; i < n; i++) {
		uint32_t j = bit_reverse(i, bits);
		if (j > i) {
			float tr = re[i]; re[i] = re[j]; re[j] = tr;
			float ti = im[i]; im[i] = im[j]; im[j] = ti;
		}
	}

	/* Cooley-Tukey butterfly */
	for (uint32_t len = 2; len <= n; len <<= 1) {
		float ang = -2.0f * (float)M_PI / (float)len;
		float wr  = cosf(ang);
		float wi  = sinf(ang);

		for (uint32_t i = 0; i < n; i += len) {
			float cr = 1.0f, ci = 0.0f;
			for (uint32_t j = 0; j < len / 2; j++) {
				uint32_t a = i + j;
				uint32_t b = i + j + len / 2;
				float tr = cr * re[b] - ci * im[b];
				float ti = cr * im[b] + ci * re[b];
				re[b] = re[a] - tr;
				im[b] = im[a] - ti;
				re[a] += tr;
				im[a] += ti;
				float ncr = cr * wr - ci * wi;
				ci = cr * wi + ci * wr;
				cr = ncr;
			}
		}
	}
}
