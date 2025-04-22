#ifndef LAZY_BUTTERFLY_FFT_64
#define LAZY_BUTTERFLY_FFT_64

#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"
#include "flint/profiler.h"

/*
void avx2_mulhi_split_lazy(__m256i* high, __m256i a, __m256i b) ;
void avx2_mullo_split_lazy(__m256i* low, __m256i a, __m256i b) ;
void avx512_mulhi_split_lazy(__m512i* high, __m512i a, __m512i b) ;
*/

void preinv_fft_lazy44(nn_ptr a, nn_ptr b, ulong w, ulong w_pr, slong len, ulong n, ulong n2, ulong p_hi, ulong p_lo, ulong tmp) ;
void avx2_preinv_split_fft_lazy44(nn_ptr a, nn_ptr b, ulong w, ulong w_pr, slong len, ulong n, ulong n2, ulong p_hi, ulong p_lo, ulong tmp) ;
void avx512_preinv_split_fft_lazy44(nn_ptr a, nn_ptr b, ulong w, ulong w_pr, slong len, ulong n, ulong n2, ulong p_hi, ulong p_lo, ulong tmp) ;


#endif
