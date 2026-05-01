#pragma once
#include <stdint.h>

/* In-place radix-2 Cooley-Tukey FFT.  n must be a power of two. */
void fft_forward(float *re, float *im, uint32_t n);
