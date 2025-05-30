#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <time.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"
#include "flint/machine_vectors.h"

#define SPLIT 20
#define MASK ((1L << SPLIT) - 1)

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

void split_dot_product_old(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
{
    ulong alo, ahi, blo, bhi;
    ulong rlo=0, rmid=0, rhi=0;

    for (slong i=0; i < len; i++)
    {
        alo = vec1[i] & MASK; //((1L << SPLIT) - 1);
        ahi = vec1[i] >> SPLIT;
        blo = vec2[i] & MASK; //((1L << SPLIT) - 1);
        bhi = vec2[i] >> SPLIT;

        rlo += alo*blo;
        rhi += ahi*bhi;
        rmid += alo*bhi + ahi*blo;
    }

    *res = rlo + (rmid << SPLIT) + (rhi << 2*SPLIT);
}

void split_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
//void split_dot_product_unroll(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
{
    __m256i alo, ahi, blo, bhi;
    __m256i v_rlo = _mm256_setzero_si256();
    __m256i v_rmi = _mm256_setzero_si256();
    __m256i v_rhi = _mm256_setzero_si256();
    const __m256i vMASK = _mm256_set1_epi64x(MASK);

    slong i = 0;

    for (; i+3 < len; i+=4)
    {
        __m256i v1 = _mm256_loadu_si256((__m256i*) (vec1+i));
        __m256i v2 = _mm256_loadu_si256((__m256i*) (vec2+i));

        alo = _mm256_and_si256(v1, vMASK);
        ahi = _mm256_srli_epi64(v1, SPLIT);
        blo = _mm256_and_si256(v2, vMASK);
        bhi = _mm256_srli_epi64(v2, SPLIT);

        v_rlo = _mm256_add_epi64(v_rlo, _mm256_mul_epu32(alo, blo));
        v_rhi = _mm256_add_epi64(v_rhi, _mm256_mul_epu32(ahi, bhi));
        v_rmi = _mm256_add_epi64(v_rmi, _mm256_mul_epu32(alo, bhi));
        v_rmi = _mm256_add_epi64(v_rmi, _mm256_mul_epu32(ahi, blo));
    }

    // gather results
    ulong rlo = vec4n_horizontal_sum(v_rlo);
    ulong rmi = vec4n_horizontal_sum(v_rmi);
    ulong rhi = vec4n_horizontal_sum(v_rhi);

    // handle extra terms if len not multiple of 4
    for (; i < len; i++)
    {
        ulong alo = vec1[i] & MASK; //((1L << SPLIT) - 1);
        ulong ahi = vec1[i] >> SPLIT;
        ulong blo = vec2[i] & MASK; //((1L << SPLIT) - 1);
        ulong bhi = vec2[i] >> SPLIT;

        rlo += alo*blo;
        rhi += ahi*bhi;
        rmi += alo*bhi + ahi*blo;
    }

    *res = rlo + (rmi << SPLIT) + (rhi << 2*SPLIT);

    //    A ADAPTER:
//    // result: ulo + 2**26 umi + 2**52 uhi
//    // hi = (umi >> 38) + (uhi >> 12)  ||  lo = (umi << 26) + (uhi << 52) + ulo
//    add_ssaaaa(dp_hi, dp_lo, (dp_mi>>38), (dp_mi<<26), (dp_hi>>12), ((dp_hi<<52)+dp_lo));
//    ulong res;
//    NMOD2_RED2(res, dp_hi, dp_lo, mod);
}

void split_kara_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
{
    ulong alo, ahi, blo, bhi;
    ulong lolo, hihi;
    ulong rlo=0, rmid=0, rhi=0;

    for (slong i=0; i < len; i++)
    {
        alo = vec1[i] & MASK;
        ahi = vec1[i] >> SPLIT;
        blo = vec2[i] & MASK;
        bhi = vec2[i] >> SPLIT;

        lolo = alo*blo;
        hihi = ahi*bhi;

        rlo += lolo;
        rhi += hihi;
        rmid += (alo + ahi)*(blo + bhi) - lolo - hihi;
    }

    *res = rlo + (rmid << SPLIT) + (rhi << 2*SPLIT);
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

    *res = vec4n_horizontal_sum(sum);

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
    *res = vec4n_horizontal_sum(sum);

    for ( ; i < len; i++)
    {
        *res += vec1[i]*vec2[i];
    }

}

void simd2_dot_product_unrolled_16(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
{
    // loop-unrolling of simd2_scalar_vector.

    __m256i sum = _mm256_setzero_si256();


    slong i;
    for (i=0; i+15 < len; i+=16)
    {
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+ 0]), _mm256_loadu_si256((const __m256i *)&vec2[i+ 0])));
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+ 4]), _mm256_loadu_si256((const __m256i *)&vec2[i+ 4])));
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+ 8]), _mm256_loadu_si256((const __m256i *)&vec2[i+ 8])));
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+12]), _mm256_loadu_si256((const __m256i *)&vec2[i+12])));
    }

    // when len is not a multiple of 16
    for ( ; i+4 < len; i+=4)
    {
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i]), _mm256_loadu_si256((const __m256i *)&vec2[i])));
    }

    // reduce sum vector
    *res = vec4n_horizontal_sum(sum);

    for ( ; i < len; i++)
    {
        *res += vec1[i]*vec2[i];
    }

}

void simd2_dot_product_unrolled_8(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
{
    // loop-unrolling of simd2_scalar_vector.

    __m256i sum = _mm256_setzero_si256();


    slong i;
    for (i=0; i+7 < len; i+=8)
    {
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+ 0]), _mm256_loadu_si256((const __m256i *)&vec2[i+ 0])));
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i+ 4]), _mm256_loadu_si256((const __m256i *)&vec2[i+ 4])));
    }

    // when len is not a multiple of 8
    if (i+4 < len)
    {
        sum = _mm256_add_epi64(sum, _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec1[i]), _mm256_loadu_si256((const __m256i *)&vec2[i])));
        i+=4;
    }

    // reduce sum vector
    *res = vec4n_horizontal_sum(sum);

    for ( ; i < len; i++)
    {
        *res += vec1[i]*vec2[i];
    }

}

#if defined(__AVX512F__)
void simd512_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
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

    *res = _mm512_reduce_add_epi64(sum);

    // when len is not a multiple of 8
    for ( ; i < len; i++)
    {
        *res += vec1[i]*vec2[i];
    }
}

void simd512_dot_product_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
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
        sum = _mm512_add_epi64(sum, _mm512_mul_epu32(_mm512_loadu_si512((const __m512i *)&vec1[i]), _mm512_loadu_si512((const __m512i *)&vec2[i])));
    }

    // reduce sum vector
    *res = _mm512_reduce_add_epi64(sum);

    for ( ; i < len; i++)
    {
        *res += vec1[i]*vec2[i];
    }
}
#endif
