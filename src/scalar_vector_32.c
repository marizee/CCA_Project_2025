#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <time.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"


void print_reg_64(char* nom, __m256i reg) 
{
    // prints values of the register assuming they are 64 bits integers.

    printf("%s =\t", nom);
    for (slong i=0; i<4; i++) {
        printf("%lld ", reg[i]);
    }
    printf("\n");
}

__attribute__((optimize("-fno-tree-vectorize")))
void seq_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len)
{
    // computes scalar-vector product of at most 32 bits integers.
    for (slong i=0; i < len; i++)
    {
        res[i] = b*vec[i];
    }
}

void seq_scalar_vector_vectorized(nn_ptr res, ulong b, nn_ptr vec, slong len)
{
    // computes scalar-vector product of at most 32 bits integers with auto-vectorization.
    for (slong i=0; i < len; i++)
    {
        res[i] = b*vec[i];
    }
}

void seq_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len)
{
    // loop-unrolling of seq_scalar_vector.

    slong i;
    for (i=0; i+3 < len; i += 4)
    {
        res[i+0] = b*vec[i+0];
        res[i+1] = b*vec[i+1];
        res[i+2] = b*vec[i+2];
        res[i+3] = b*vec[i+3];
    }

    // when len is not a multiple of 4
    for ( ; i < len; i++)
    {
        res[i] = b*vec[i];
    }
}

// TODO: specify `-mno-avx512f` (uses avx2, but in case of)
void simd2_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len)
{
    // computes scalar-vector product of at most 32 bits integers using intrinsics.
    
    __m256i vb = _mm256_set1_epi64x((int)b);

    slong i;
    for (i=0; i+3 < len; i+=4)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)&vec[i]);
        __m256i prod = _mm256_mul_epu32(vb, va);
        _mm256_storeu_si256((__m256i *)&res[i], prod);
    }

    // when len is not a multiple of 4
    for ( ; i < len; i++)
    {
        res[i] = b*vec[i];
    }
}

void simd2_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len)
{
    // loop-unrolling of simd2_scalar_vector.

    __m256i vb = _mm256_set1_epi64x((int)b);

    slong i;    
    for (i=0; i+31 < len; i+=32)
    {
        _mm256_storeu_si256((__m256i *)&res[i+ 0], _mm256_mul_epu32(vb, _mm256_loadu_si256((const __m256i *)&vec[i+ 0])));
        _mm256_storeu_si256((__m256i *)&res[i+ 4], _mm256_mul_epu32(vb, _mm256_loadu_si256((const __m256i *)&vec[i+ 4])));
        _mm256_storeu_si256((__m256i *)&res[i+ 8], _mm256_mul_epu32(vb, _mm256_loadu_si256((const __m256i *)&vec[i+ 8])));
        _mm256_storeu_si256((__m256i *)&res[i+12], _mm256_mul_epu32(vb, _mm256_loadu_si256((const __m256i *)&vec[i+12])));
        _mm256_storeu_si256((__m256i *)&res[i+16], _mm256_mul_epu32(vb, _mm256_loadu_si256((const __m256i *)&vec[i+16])));
        _mm256_storeu_si256((__m256i *)&res[i+20], _mm256_mul_epu32(vb, _mm256_loadu_si256((const __m256i *)&vec[i+20])));
        _mm256_storeu_si256((__m256i *)&res[i+24], _mm256_mul_epu32(vb, _mm256_loadu_si256((const __m256i *)&vec[i+24])));
        _mm256_storeu_si256((__m256i *)&res[i+28], _mm256_mul_epu32(vb, _mm256_loadu_si256((const __m256i *)&vec[i+28])));
    }
    
    // when len is not a multiple of 32
    for ( ; i+4 < len; i+=4)
    {
        _mm256_storeu_si256((__m256i *)&res[i], _mm256_mul_epu32(vb, _mm256_loadu_si256((const __m256i *)&vec[i])));   
    }

    for ( ; i < len; i++)
    {
        res[i] = b*vec[i];
    }
}

#if defined(__AVX512F__)
void simd512_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len)
{
    __m512i vb = _mm512_set1_epi64(b);

    slong i;
    for (i=0; i+7 < len; i+=8)
    {
        __m512i va = _mm512_loadu_si512((const __m512i *)&vec[i]);
        __m512i prod = _mm512_mul_epu32(vb, va);
        _mm512_storeu_si512 ((__m512i *)&res[i], prod);
    }

    // when len is not a multiple of 8
    for ( ; i < len; i++)
    {
        res[i] = b*vec[i];
    }
}

void simd512_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len)
{
    __m512i vb = _mm512_set1_epi64(b);

    slong i; 
    for (i=0; i+31 < len; i+=32)
    {
        _mm512_storeu_si512((__m512i *)&res[i+ 0], _mm512_mul_epu32(vb, _mm512_loadu_si512((const __m512i *)&vec[i+ 0])));
        _mm512_storeu_si512((__m512i *)&res[i+ 8], _mm512_mul_epu32(vb, _mm512_loadu_si512((const __m512i *)&vec[i+ 8])));
        _mm512_storeu_si512((__m512i *)&res[i+16], _mm512_mul_epu32(vb, _mm512_loadu_si512((const __m512i *)&vec[i+16])));
        _mm512_storeu_si512((__m512i *)&res[i+24], _mm512_mul_epu32(vb, _mm512_loadu_si512((const __m512i *)&vec[i+24])));
    }

    // when len is not a multiple of 32
    for ( ; i+7 < len; i+=8)
    {
        _mm512_storeu_si512((__m512i *)&res[i], _mm512_mul_epu32(vb, _mm512_loadu_si512((const __m512i *)&vec[i])));
    }

    for ( ; i < len; i++)
    {
        res[i] = b*vec[i];
    }
}
#endif