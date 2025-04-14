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

#define SPLIT 20
#define MASK ((1L << SPLIT) - 1)

__attribute__((optimize("-fno-tree-vectorize")))
void seq_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    // computes the dot product of vectors with at most 32 bits integers.
    *res=0;
    for (slong i=0; i < len; i++)
        *res += vec1[i]*vec2[i];
    NMOD_RED(*res, *res, mod);
}

void seq_dot_product_mod_vectorized(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    // computes the dot product of vectors with at most 32 bits integers.
    *res=0;
    for (slong i=0; i < len; i++)
        *res += vec1[i]*vec2[i];
    NMOD_RED(*res, *res, mod);
}


void seq_dot_product_mod_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;

void split_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
//void split_dot_product_unroll(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
{
    ulong rlo = 0;
    ulong rmi = 0;
    ulong rhi = 0;

    for (slong i=0; i < len; i++)
    {
        ulong alo = vec1[i] & MASK; //((1L << SPLIT) - 1);
        ulong ahi = vec1[i] >> SPLIT;
        ulong blo = vec2[i] & MASK; //((1L << SPLIT) - 1);
        ulong bhi = vec2[i] >> SPLIT;

        rlo += alo*blo;
        rhi += ahi*bhi;
        rmi += alo*bhi + ahi*blo;
    }

    add_ssaaaa(rhi, rlo, (rmi>>(64-SPLIT)), (rmi<<SPLIT), (rhi>>(64-2*SPLIT)), ((rhi<<(2*SPLIT))+rlo));
    NMOD2_RED2(*res, rhi, rlo, mod);
}

void split_kara_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
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

        rlo = rlo + lolo;
        rhi = rhi + hihi;
        rmid += (alo + ahi)*(blo + bhi) - lolo - hihi;
    }

    add_ssaaaa(rhi, rlo, (rmid>>(64-SPLIT)), (rmid<<SPLIT), (rhi>>(64-2*SPLIT)), ((rhi<<(2*SPLIT))+rlo));
    NMOD2_RED2(*res, rhi, rlo, mod);
}

// Doesn't work for int size > 15 bits. No mul64 => need split
void simd2_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    // computes dot product of vectors with at most 32 bits integers using intrinsics.

    __m256i sum = _mm256_setzero_si256(); 
    __m256i vmod = _mm256_set1_epi64x((int)mod.n);
    __m256 vqinv = _mm256_set1_ps(1.0/mod.n);

    slong i;
    for (i=0; i+3 < len; i+=4)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)&vec1[i]);
        __m256i vb = _mm256_loadu_si256((const __m256i *)&vec2[i]);
        __m256i prod = _mm256_mul_epu32(va, vb);

        __m256 vtmp = _mm256_mul_ps(_mm256_cvtepi32_ps(prod), vqinv);
        __m256i vtmp2 = _mm256_mul_epi32(_mm256_cvtps_epi32(vtmp), vmod);
        __m256i masked = _mm256_and_si256(vmod, _mm256_cmpgt_epi32(vtmp2, prod));

        vtmp2 = _mm256_sub_epi32(prod, _mm256_sub_epi32(vtmp2, masked));
        sum = _mm256_add_epi64(sum, vtmp2);

        vtmp2 = _mm256_or_si256(_mm256_cmpgt_epi64(sum, vmod), _mm256_cmpeq_epi64(sum, vmod));
        masked = _mm256_and_si256(vtmp2, vmod);
        sum = _mm256_sub_epi64(sum, masked);
    }

    *res = vec4n_horizontal_sum(sum);
    for (slong i=0; i < len; i++)
        *res += vec1[i]*vec2[i];
    NMOD_RED(*res, *res, mod);
}

void simd2_split_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
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

    add_ssaaaa(rhi, rlo, (rmi>>(64-SPLIT)), (rmi<<SPLIT), (rhi>>(64-2*SPLIT)), ((rhi<<(2*SPLIT))+rlo));
    NMOD2_RED2(*res, rhi, rlo, mod);

    // ulong lo_mask = ((1l << FLINT_BITS) - 1);
    // ulong tmp_hi, tmp_lo, tmp_acc;
    // tmp_hi= rhi >> (FLINT_BITS - SPLIT);
    // tmp_lo = ((rhi << SPLIT) & lo_mask) + rmi;
    // NMOD_RED2(tmp_acc, tmp_hi, tmp_lo, mod);
    // tmp_hi = tmp_acc >> (FLINT_BITS - SPLIT);
    // tmp_lo = ((tmp_acc << SPLIT) & lo_mask) + rlo;
    // NMOD_RED2(*res, tmp_hi, tmp_lo, mod);

    // *res = rlo + (rmi << SPLIT) + (rhi << 2*SPLIT);
}

void simd2_kara_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    __m256i v_rlo = _mm256_setzero_si256();
    __m256i v_rmi = _mm256_setzero_si256();
    __m256i v_rhi = _mm256_setzero_si256();
    const __m256i vMASK = _mm256_set1_epi64x(MASK);

    slong i = 0;

    for (; i+3 < len; i+=4)
    {
        __m256i v1 = _mm256_loadu_si256((__m256i*) (vec1+i));
        __m256i v2 = _mm256_loadu_si256((__m256i*) (vec2+i));

        __m256i alo = _mm256_and_si256(v1, vMASK);
        __m256i ahi = _mm256_srli_epi64(v1, SPLIT);
        __m256i blo = _mm256_and_si256(v2, vMASK);
        __m256i bhi = _mm256_srli_epi64(v2, SPLIT);

        __m256i hihi = _mm256_mul_epu32(ahi, bhi);
        __m256i lolo = _mm256_mul_epu32(alo, blo);

        __m256i asum = _mm256_add_epi64(alo, ahi);
        __m256i bsum = _mm256_add_epi64(blo, bhi);

        v_rlo = _mm256_add_epi64(v_rlo, lolo);
        v_rhi = _mm256_add_epi64(v_rhi, hihi);
        v_rmi = _mm256_add_epi64(v_rmi, _mm256_mul_epu32(asum, bsum));
        v_rmi = _mm256_sub_epi64(v_rmi, _mm256_add_epi64(hihi, lolo));
    }

    // gather results
    ulong rlo = vec4n_horizontal_sum(v_rlo);
    ulong rmi = vec4n_horizontal_sum(v_rmi);
    ulong rhi = vec4n_horizontal_sum(v_rhi);

    for (; i < len; i++)
    {
        ulong alo = vec1[i] & MASK;
        ulong ahi = vec1[i] >> SPLIT;
        ulong blo = vec2[i] & MASK;
        ulong bhi = vec2[i] >> SPLIT;

        ulong lolo = alo*blo;
        ulong hihi = ahi*bhi;

        rlo += lolo;
        rhi += hihi;
        rmi += (alo + ahi)*(blo + bhi) - lolo - hihi;
    }

    add_ssaaaa(rhi, rlo, (rmi>>(64-SPLIT)), (rmi<<SPLIT), (rhi>>(64-2*SPLIT)), ((rhi<<(2*SPLIT))+rlo));
    NMOD2_RED2(*res, rhi, rlo, mod);
}

void simd2_dot_product_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;

#if defined(__AVX512F__)
void simd512_split_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) {
    __m512i vrlo = _mm512_setzero_si512();
    __m512i vrmi = _mm512_setzero_si512();
    __m512i vrhi = _mm512_setzero_si512();
    const __m512i vMASK = _mm512_set1_epi64(MASK);

    slong i = 0;

    for(; i+7 < len; i+=8)
    {
        __m512i v1 = _mm512_loadu_si512((const __m512i *)&vec1[i]);
        __m512i v2 = _mm512_loadu_si512((const __m512i *)&vec2[i]);

        __m512i alo = _mm512_and_si512(v1, vMASK);
        __m512i ahi = _mm512_srli_epi64(v1, SPLIT);
        __m512i blo = _mm512_and_si512(v2, vMASK);
        __m512i bhi = _mm512_srli_epi64(v2, SPLIT);

        __m512i hihi = _mm512_mul_epu32(ahi, bhi);
        __m512i lolo = _mm512_mul_epu32(alo, blo);

        vrlo = _mm512_add_epi64(vrlo, lolo);
        vrhi = _mm512_add_epi64(vrhi, hihi);
        vrmi = _mm512_add_epi64(vrmi, _mm512_mul_epu32(alo, bhi));
        vrmi = _mm512_add_epi64(vrmi, _mm512_mul_epu32(ahi, blo));
    }

    ulong rlo = _mm512_reduce_add_epi64(vrlo);
    ulong rmi = _mm512_reduce_add_epi64(vrmi);
    ulong rhi = _mm512_reduce_add_epi64(vrhi);

    for(; i < len; i++)
    {
        ulong alo = vec1[i] & MASK; //((1L << SPLIT) - 1);
        ulong ahi = vec1[i] >> SPLIT;
        ulong blo = vec2[i] & MASK; //((1L << SPLIT) - 1);
        ulong bhi = vec2[i] >> SPLIT;

        rlo += alo*blo;
        rhi += ahi*bhi;
        rmi += alo*bhi + ahi*blo;
    }

    add_ssaaaa(rhi, rlo, (rmi>>(64-SPLIT)), (rmi<<SPLIT), (rhi>>(64-2*SPLIT)), ((rhi<<(2*SPLIT))+rlo));
    NMOD2_RED2(*res, rhi, rlo, mod);
}
void simd512_kara_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) {
    __m512i vrlo = _mm512_setzero_si512();
    __m512i vrmi = _mm512_setzero_si512();
    __m512i vrhi = _mm512_setzero_si512();
    const __m512i vMASK = _mm512_set1_epi64(MASK);

    slong i = 0;

    for(; i+7 < len; i+=8)
    {
        __m512i v1 = _mm512_loadu_si512((const __m512i *)&vec1[i]);
        __m512i v2 = _mm512_loadu_si512((const __m512i *)&vec2[i]);

        __m512i alo = _mm512_and_si512(v1, vMASK);
        __m512i ahi = _mm512_srli_epi64(v1, SPLIT);
        __m512i blo = _mm512_and_si512(v2, vMASK);
        __m512i bhi = _mm512_srli_epi64(v2, SPLIT);

        __m512i hihi = _mm512_mul_epu32(ahi, bhi);
        __m512i lolo = _mm512_mul_epu32(alo, blo);

        __m512i asum = _mm512_add_epi64(alo, ahi);
        __m512i bsum = _mm512_add_epi64(blo, bhi);

        vrlo = _mm512_add_epi64(vrlo, lolo);
        vrhi = _mm512_add_epi64(vrhi, hihi);
        vrmi = _mm512_add_epi64(vrmi, _mm512_mul_epu32(asum, bsum));
        vrmi = _mm512_sub_epi64(vrmi, _mm512_add_epi64(hihi, lolo));
    }

    ulong rlo = _mm512_reduce_add_epi64(vrlo);
    ulong rmi = _mm512_reduce_add_epi64(vrmi);
    ulong rhi = _mm512_reduce_add_epi64(vrhi);

    for(; i < len; i++)
    {
        ulong alo = vec1[i] & MASK; //((1L << SPLIT) - 1);
        ulong ahi = vec1[i] >> SPLIT;
        ulong blo = vec2[i] & MASK; //((1L << SPLIT) - 1);
        ulong bhi = vec2[i] >> SPLIT;

        rlo += alo*blo;
        rhi += ahi*bhi;
        rmi += alo*bhi + ahi*blo;
    }

    add_ssaaaa(rhi, rlo, (rmi>>(64-SPLIT)), (rmi<<SPLIT), (rhi>>(64-2*SPLIT)), ((rhi<<(2*SPLIT))+rlo));
    NMOD2_RED2(*res, rhi, rlo, mod);
}
#endif
