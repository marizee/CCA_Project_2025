#ifndef MULSPLIT_H
#define MULSPLIT_H

#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"
#include "flint/profiler.h"

ulong mulhi_split(ulong a, ulong b) ;
ulong mullo_split(ulong a, ulong b) ;
void mul_split(ulong hi, ulong lo, ulong a, ulong b) ;

__m256i avx2_mulhi_split(__m256i a, __m256i b) ;
__m256i avx2_mullo_split(__m256i a, __m256i b) ;
__m256i avx2_mullo_epi64(__m256i a, __m256i b) ;
void avx2_mul_split(__m256i* hi, __m256i* lo, __m256i a, __m256i b) ;

#if defined(__AVX512F__)
__m512i avx512_mulhi_split(__m512i a, __m512i b) ;
__m512i avx512_mul_split(__m512i hi, __m512i lo, __m512i a, __m512i b) ;
#endif

#endif
