#ifndef LAZY_BUTTERFLY_FFT_64
#define LAZY_BUTTERFLY_FFT_64

#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"
#include "flint/profiler.h"


void preinv_fft_lazy44(nn_ptr a, nn_ptr b, ulong w, ulong w_pr, slong len, ulong n, ulong n2, ulong p_hi, ulong p_lo, ulong tmp) ;
void avx2_preinv_split_fft_lazy44(nn_ptr a, nn_ptr b, ulong w, ulong w_pr, slong len, ulong n, ulong n2, ulong p_hi, ulong p_lo, ulong tmp) ;
void avx512_preinv_split_fft_lazy44(nn_ptr a, nn_ptr b, ulong w, ulong w_pr, slong len, ulong n, ulong n2, ulong p_hi, ulong p_lo, ulong tmp) ;


#endif
