#include "dot_prod_mod_64.h"
#include "mulsplit.h"

#include "flint/machine_vectors.h"

#define MASK ((1L << 32) - 1)

#define SPLIT 20
#define SMASK ((1L << SPLIT) - 1)

// TODO : DELAYED REDUCTION IF POSSIBLE

void flint_dot_prod_mod_64(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    dot_params_t params = _nmod_vec_dot_params(len, mod);
    *res = _nmod_vec_dot(a, b, len, mod, params);
}

__attribute__((optimize("-fno-tree-vectorize")))
void seq_dot_prod_mod_64(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    ulong t_hi=0, t_mi=0, t_lo=0;
    ulong q_hi, q_lo;

    for (slong i = 0; i < len; i++)
    {
        umul_ppmm(q_hi, q_lo, a[i], b[i]);
        add_sssaaaaaa(t_hi, t_mi, t_lo, t_hi, t_mi, t_lo, 0, q_hi, q_lo);
    }
    NMOD_RED(t_hi, t_hi, mod);
    NMOD_RED3(*res, t_hi, t_mi, t_lo, mod);
}

void seq_dot_prod_mod_64_vectorized(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    ulong t_hi=0, t_mi=0, t_lo=0;
    ulong q_hi, q_lo;

    for (slong i = 0; i < len; i++)
    {
        umul_ppmm(q_hi, q_lo, a[i], b[i]);
        add_sssaaaaaa(t_hi, t_mi, t_lo, t_hi, t_mi, t_lo, 0, q_hi, q_lo);
    }
    NMOD_RED(t_hi, t_hi, mod);
    NMOD_RED3(*res, t_hi, t_mi, t_lo, mod);
}

void seq_dot_prod_mod_64_unrolled(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    ulong t_hi=0, t_mi=0, t_lo=0;
    ulong q_hi, q_lo;

    for (slong i = 0; i+3 < len; i+=4)
    {
        umul_ppmm(q_hi, q_lo, a[i+0], b[i+0]);
        add_sssaaaaaa(t_hi, t_mi, t_lo, t_hi, t_mi, t_lo, 0, q_hi, q_lo);
        umul_ppmm(q_hi, q_lo, a[i+1], b[i+1]);
        add_sssaaaaaa(t_hi, t_mi, t_lo, t_hi, t_mi, t_lo, 0, q_hi, q_lo);
        umul_ppmm(q_hi, q_lo, a[i+2], b[i+2]);
        add_sssaaaaaa(t_hi, t_mi, t_lo, t_hi, t_mi, t_lo, 0, q_hi, q_lo);
        umul_ppmm(q_hi, q_lo, a[i+3], b[i+3]);
        add_sssaaaaaa(t_hi, t_mi, t_lo, t_hi, t_mi, t_lo, 0, q_hi, q_lo);
    }
    NMOD_RED(t_hi, t_hi, mod);
    NMOD_RED3(*res, t_hi, t_mi, t_lo, mod);
}

void split_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
//void split_dot_product_unroll(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len)
{
    ulong rlo = 0;
    ulong rmi = 0;
    ulong rhi = 0;

    for (slong i=0; i < len; i++)
    {
        ulong alo = vec1[i] & SMASK; //((1L << SPLIT) - 1);
        ulong ahi = vec1[i] >> SPLIT;
        ulong blo = vec2[i] & SMASK; //((1L << SPLIT) - 1);
        ulong bhi = vec2[i] >> SPLIT;

        rlo += alo*blo;
        rhi += ahi*bhi;
        rmi += alo*bhi + ahi*blo;
    }

    add_ssaaaa(rhi, rlo, (rhi>>(64-2*SPLIT)), rhi<<(2*SPLIT), 0, rlo);
    add_ssaaaa(rhi, rlo, (rmi>>(64-SPLIT)), (rmi<<SPLIT), rhi, rlo);
    NMOD2_RED2(*res, rhi, rlo, mod);
}

void split_kara_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    ulong alo, ahi, blo, bhi;
    ulong lolo, hihi;
    ulong rlo=0, rmid=0, rhi=0;

    for (slong i=0; i < len; i++)
    {
        alo = vec1[i] & SMASK;
        ahi = vec1[i] >> SPLIT;
        blo = vec2[i] & SMASK;
        bhi = vec2[i] >> SPLIT;

        lolo = alo*blo;
        hihi = ahi*bhi;

        rlo = rlo + lolo;
        rhi = rhi + hihi;
        rmid += (alo + ahi)*(blo + bhi) - lolo - hihi;
    }

    add_ssaaaa(rhi, rlo, (rhi>>(64-2*SPLIT)), rhi<<(2*SPLIT), 0, rlo);
    add_ssaaaa(rhi, rlo, (rmid>>(64-SPLIT)), (rmid<<SPLIT), rhi, rlo);
    NMOD2_RED2(*res, rhi, rlo, mod);
}


void simd2_dot_prod_mod_64(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    //__m256i vt_hi = _mm256_setzero_si256();
    //__m256i vt_mi = _mm256_setzero_si256();
    //__m256i vt_lo = _mm256_setzero_si256();

    __m256i vq_hi = _mm256_setzero_si256();
    __m256i vq_lo = _mm256_setzero_si256();

    slong i;
    for (i = 0; i+3 < len; i += 4)
    {
        __m256i va = _mm256_loadu_si256((__m256i*)(a+i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b+i));

        // umulppmm-avx2
        avx2_mul_split(&vq_hi, &vq_lo, va, vb);

        // TODO
    }
}

void simd2_split_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    __m256i alo, ahi, blo, bhi;
    __m256i v_rlo = _mm256_setzero_si256();
    __m256i v_rmi = _mm256_setzero_si256();
    __m256i v_rhi = _mm256_setzero_si256();
    const __m256i vMASK = _mm256_set1_epi64x(SMASK);

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
        ulong alo = vec1[i] & SMASK; //((1L << SPLIT) - 1);
        ulong ahi = vec1[i] >> SPLIT;
        ulong blo = vec2[i] & SMASK; //((1L << SPLIT) - 1);
        ulong bhi = vec2[i] >> SPLIT;

        rlo += alo*blo;
        rhi += ahi*bhi;
        rmi += alo*bhi + ahi*blo;
    }

    add_ssaaaa(rhi, rlo, (rhi>>(64-2*SPLIT)), rhi<<(2*SPLIT), 0, rlo);
    add_ssaaaa(rhi, rlo, (rmi>>(64-SPLIT)), (rmi<<SPLIT), rhi, rlo);
    NMOD2_RED2(*res, rhi, rlo, mod);
}

void simd2_kara_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod)
{
    __m256i v_rlo = _mm256_setzero_si256();
    __m256i v_rmi = _mm256_setzero_si256();
    __m256i v_rhi = _mm256_setzero_si256();
    const __m256i vMASK = _mm256_set1_epi64x(SMASK);

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
        ulong alo = vec1[i] & SMASK;
        ulong ahi = vec1[i] >> SPLIT;
        ulong blo = vec2[i] & SMASK;
        ulong bhi = vec2[i] >> SPLIT;

        ulong lolo = alo*blo;
        ulong hihi = ahi*bhi;

        rlo += lolo;
        rhi += hihi;
        rmi += (alo + ahi)*(blo + bhi) - lolo - hihi;
    }

    add_ssaaaa(rhi, rlo, (rhi>>(64-2*SPLIT)), rhi<<(2*SPLIT), 0, rlo);
    add_ssaaaa(rhi, rlo, (rmi>>(64-SPLIT)), (rmi<<SPLIT), rhi, rlo);
    NMOD2_RED2(*res, rhi, rlo, mod);
}

#if defined(__AVX512F__)
void simd512_split_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) {
    __m512i vrlo = _mm512_setzero_si512();
    __m512i vrmi = _mm512_setzero_si512();
    __m512i vrhi = _mm512_setzero_si512();
    const __m512i vMASK = _mm512_set1_epi64(SMASK);

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
        ulong alo = vec1[i] & SMASK; //((1L << SPLIT) - 1);
        ulong ahi = vec1[i] >> SPLIT;
        ulong blo = vec2[i] & SMASK; //((1L << SPLIT) - 1);
        ulong bhi = vec2[i] >> SPLIT;

        rlo += alo*blo;
        rhi += ahi*bhi;
        rmi += alo*bhi + ahi*blo;
    }

    add_ssaaaa(rhi, rlo, (rhi>>(64-2*SPLIT)), rhi<<(2*SPLIT), 0, rlo);
    add_ssaaaa(rhi, rlo, (rmi>>(64-SPLIT)), (rmi<<SPLIT), rhi, rlo);
    NMOD2_RED2(*res, rhi, rlo, mod);
}
void simd512_kara_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) {
    __m512i vrlo = _mm512_setzero_si512();
    __m512i vrmi = _mm512_setzero_si512();
    __m512i vrhi = _mm512_setzero_si512();
    const __m512i vMASK = _mm512_set1_epi64(SMASK);

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
        ulong alo = vec1[i] & SMASK; //((1L << SPLIT) - 1);
        ulong ahi = vec1[i] >> SPLIT;
        ulong blo = vec2[i] & SMASK; //((1L << SPLIT) - 1);
        ulong bhi = vec2[i] >> SPLIT;

        rlo += alo*blo;
        rhi += ahi*bhi;
        rmi += alo*bhi + ahi*blo;
    }

    add_ssaaaa(rhi, rlo, (rhi>>(64-2*SPLIT)), rhi<<(2*SPLIT), 0, rlo);
    add_ssaaaa(rhi, rlo, (rmi>>(64-SPLIT)), (rmi<<SPLIT), rhi, rlo);
    NMOD2_RED2(*res, rhi, rlo, mod);
}
#endif
