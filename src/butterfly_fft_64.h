#ifndef BUTTERFLY_FFT_64
#define BUTTERFLY_FFT_64

#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"
#include "flint/profiler.h"

void seq_fft(nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod) ;
void preinv_fft(nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod) ;

void preinv_split_fft(nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod) ;
void avx2_preinv_split_fft(nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod) ;

#endif
