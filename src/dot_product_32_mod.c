#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <time.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"
#include "flint/machine_vectors.h"
#include "flint/nmod.h"

void flint_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    dot_params_t params = _nmod_vec_dot_params(len, mod);
    *res = _nmod_vec_dot(vec1, vec2, len, mod, params);
}

__attribute__((optimize("-fno-tree-vectorize")))
void seq_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    // computes the dot product of vectors with at most 32 bits integers.
    ulong rres=0;
    for (slong i=0; i < len; i++)
        rres += vec1[i]*vec2[i];
    NMOD_RED(*res, rres, mod);
}

void seq_dot_product_mod_vectorized(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    // computes the dot product of vectors with at most 32 bits integers.
    ulong rres=0;
    for (slong i=0; i < len; i++)
        rres += vec1[i]*vec2[i];
    NMOD_RED(*res, rres, mod);
}

void seq_dot_product_mod_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    // loop-unrolling of seq_dot_product.

    ulong rres=0;
    slong i;
    for (i=0; i+3 < len; i += 4)
    {
        rres += vec1[i+0]*vec2[i+0];
        rres += vec1[i+1]*vec2[i+1];
        rres += vec1[i+2]*vec2[i+2];
        rres += vec1[i+3]*vec2[i+3];
    }

    // when len is not a multiple of 4
    for ( ; i < len; i++)
        rres += vec1[i]*vec2[i];
    NMOD_RED(*res, rres, mod);
}

// Doesn't work for int size > 15 bits. No mul64 => need split
void simd2_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
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

    *res = vec4n_horizontal_sum(sum);
    for (; i < len; i++)
        *res += vec1[i]*vec2[i];
    NMOD_RED(*res, *res, mod);
}

void simd2_dot_product_mod_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
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
        __m256i va = _mm256_loadu_si256((const __m256i *)&vec1[i]);
        __m256i vb = _mm256_loadu_si256((const __m256i *)&vec2[i]);
        __m256i prod = _mm256_mul_epu32(va, vb);
        sum = _mm256_add_epi64(sum, prod);
    }

    // reduce sum vector
    ulong rres = vec4n_horizontal_sum(sum);

    // when len is not a multiple of 4
    for (; i < len; i++)
        rres += vec1[i]*vec2[i];
    NMOD_RED(*res, rres, mod);
}


#if defined(__AVX512F__)
void simd512_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    __m512i sum = _mm512_setzero_si512();

    slong i;
    for (i=0; i+7 < len; i+=8)
    {
        __m512i va = _mm512_loadu_si512((const __m512i *)&vec1[i]);
        __m512i vb = _mm512_loadu_si512((const __m512i *)&vec2[i]);
        __m512i prod = _mm512_mul_epu32(va, vb);

        sum = _mm512_add_epi64(sum, prod);
    }

    ulong rres = _mm512_reduce_add_epi64(sum);

    // when len is not a multiple of 8
    for (; i < len; i++)
        rres += vec1[i]*vec2[i];
    NMOD_RED(*res, rres, mod);
}

void simd512_dot_product_mod_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    __m512i sum = _mm512_setzero_si512();

    slong i;
    for (i=0; i+31 < len; i+=32)
    {
        sum = _mm512_add_epi64(sum, _mm512_mul_epu32(_mm512_loadu_si512((const __m512i *)&vec1[i+ 0]), _mm512_loadu_si512((const __m512i *)&vec2[i+ 0])));
        sum = _mm512_add_epi64(sum, _mm512_mul_epu32(_mm512_loadu_si512((const __m512i *)&vec1[i+ 8]), _mm512_loadu_si512((const __m512i *)&vec2[i+ 8])));
        sum = _mm512_add_epi64(sum, _mm512_mul_epu32(_mm512_loadu_si512((const __m512i *)&vec1[i+16]), _mm512_loadu_si512((const __m512i *)&vec2[i+16])));
        sum = _mm512_add_epi64(sum, _mm512_mul_epu32(_mm512_loadu_si512((const __m512i *)&vec1[i+24]), _mm512_loadu_si512((const __m512i *)&vec2[i+24])));
    }

    // when len is not a multiple of 32
    for ( ; i+7 < len; i+=8)
    {
        __m512i va = _mm512_loadu_si512((const __m512i *)&vec1[i]);
        __m512i vb = _mm512_loadu_si512((const __m512i *)&vec2[i]);
        __m512i prod = _mm512_mul_epu32(va, vb);

        sum = _mm512_add_epi64(sum, prod);
    }

    ulong rres = _mm512_reduce_add_epi64(sum);

    // when len is not a multiple of 8
    for (; i < len; i++)
        rres += vec1[i]*vec2[i];
    NMOD_RED(*res, rres, mod);
}
#endif
