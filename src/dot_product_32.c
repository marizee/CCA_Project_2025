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
void seq_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
{
    // computes the dot product of vectors with at most 32 bits integers.
    for (slong i=0; i < len; i++)
    {
        *res += vec1[i]*vec2[i];
    }
}

void seq_dot_product_vectorized(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
{
    // computes the dot product of vectors with at most 32 bits integers with auto-vectorization.
    for (slong i=0; i < len; i++)
    {
        *res += vec1[i]*vec2[i];
    }
}

void seq_dot_product_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
{
    // loop-unrolling of seq_dot_product.
    
    slong i;
    for (i=0; i+3 < len; i += 4)
    {
        *res += vec1[i+0]*vec2[i+0];
        *res += vec1[i+1]*vec2[i+1];
        *res += vec1[i+2]*vec2[i+2];
        *res += vec1[i+3]*vec2[i+3];
    }

    // when len is not a multiple of 4
    for ( ; i < len; i++)
    {
        *res += vec1[i]*vec2[i];
    }
}

void simd2_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
{
    // computes dot product of vectors with at most 32 bits integers using intrinsics.
    
    __m256i sum = _mm256_setzero_si256();

    slong i;
    for (i=0; i+3 < len; i+=4)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)&vec1[i]);
        __m256i vb = _mm256_loadu_si256((const __m256i *)&vec2[i]);
        __m256i prod = _mm256_mul_epu32(va, vb);
        sum = _mm256_add_epi64(sum, prod);
    }

    // reduce sum vector
    sum = (__m256i)_mm256_hadd_pd((__m256d)sum, (__m256d)sum);
    *res = (ulong)sum[0] + (ulong)sum[2];

    // when len is not a multiple of 4
    for ( ; i < len; i++)
    {
        *res += vec1[i]*vec2[i];
    }
}

void simd2_dot_product_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
{
    // loop-unrolling of simd2_scalar_vector.

    __m256i sum = _mm256_setzero_si256();


    slong i;    
    for (i=0; i+31 < len; i+=32)
    {
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+ 0]), _mm256_loadu_si256((const __m256i *)&vec2[i+ 0])));
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+ 4]), _mm256_loadu_si256((const __m256i *)&vec2[i+ 4])));
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+ 8]), _mm256_loadu_si256((const __m256i *)&vec2[i+ 8])));
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+12]), _mm256_loadu_si256((const __m256i *)&vec2[i+12])));
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+16]), _mm256_loadu_si256((const __m256i *)&vec2[i+16])));
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+20]), _mm256_loadu_si256((const __m256i *)&vec2[i+20])));
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+24]), _mm256_loadu_si256((const __m256i *)&vec2[i+24])));
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+28]), _mm256_loadu_si256((const __m256i *)&vec2[i+28])));
    }

    // when len is not a multiple of 32
    for ( ; i+4 < len; i+=4)
    {
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i]), _mm256_loadu_si256((const __m256i *)&vec2[i])));
    }

    // reduce sum vector
    sum = (__m256i)_mm256_hadd_pd((__m256d)sum, (__m256d)sum);
    *res = (ulong)sum[0] + (ulong)sum[2];

    for ( ; i < len; i++)
    {
        *res += vec1[i]*vec2[i];
    }
}

#if defined(__AVX512F__)
void simd512_dot_product(nn_ptr res, ulong b, nn_ptr vec, slong len)
{

}

void simd512_dot_product_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len)
{

}
#endif
