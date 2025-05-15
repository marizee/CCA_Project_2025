#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>

#include "scalar_vector_mod_64.h"
#include "mulsplit.h"

void flint_shoup_scalar_vector_mod_64(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod)
{
    _nmod_vec_scalar_mul_nmod_shoup(res, b, len, w, mod);
}

__attribute__((optimize("-fno-tree-vectorize")))
void seq_shoup_scalar_vector_mod_64(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod)
{
    ulong p_hi=0, p_lo=0;
    ulong tmp;

    ulong w_pr = n_mulmod_precomp_shoup(w, mod.n);

    for (slong i = 0; i < len; i++)
    {
        umul_ppmm(p_hi, p_lo, b[i], w_pr);
        tmp = w*b[i] - p_hi*mod.n;
        if (tmp >= mod.n)
            tmp = tmp - mod.n;
        res[i] = tmp;
    }
}

void seq_shoup_scalar_vector_mod_64_vectorized(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod)
{
    ulong p_hi=0, p_lo=0;
    ulong tmp;

    ulong w_pr = n_mulmod_precomp_shoup(w, mod.n);

    for (slong i = 0; i < len; i++)
    {
        umul_ppmm(p_hi, p_lo, b[i], w_pr);
        tmp = w*b[i] - p_hi*mod.n;
        if (tmp >= mod.n)
            tmp = tmp - mod.n;
        res[i] = tmp;
    }
}

void seq_shoup_scalar_vector_mod_64_unrolled(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod)
{
    ulong p_hi=0, p_lo=0;
    ulong tmp;

    ulong w_pr = n_mulmod_precomp_shoup(w, mod.n);

    slong i;
    for (i = 0; i+3 < len; i += 4)
    {
        umul_ppmm(p_hi, p_lo, b[i+0], w_pr);
        tmp = w*b[i+0] - p_hi*mod.n;
        res[i+0] = (tmp >= mod.n) ? tmp - mod.n : tmp;
        umul_ppmm(p_hi, p_lo, b[i+1], w_pr);
        tmp = w*b[i+1] - p_hi*mod.n;
        res[i+1] = (tmp >= mod.n) ? tmp - mod.n : tmp;
        umul_ppmm(p_hi, p_lo, b[i+2], w_pr);
        tmp = w*b[i+2] - p_hi*mod.n;
        res[i+2] = (tmp >= mod.n) ? tmp - mod.n : tmp;
        umul_ppmm(p_hi, p_lo, b[i+3], w_pr);
        tmp = w*b[i+3] - p_hi*mod.n;
        res[i+3] = (tmp >= mod.n) ? tmp - mod.n : tmp;
    }

    for ( ; i < len; i++)
    {
        umul_ppmm(p_hi, p_lo, b[i], w_pr);
        tmp = w*b[i] - p_hi*mod.n;
        res[i] = (tmp >= mod.n) ? tmp - mod.n : tmp;
    }
}


void avx2_shoup_scalar_vector_mod_64(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod)
{
    ulong w_pr = n_mulmod_precomp_shoup(w, mod.n);

    __m256i vw = _mm256_set1_epi64x(w);
    __m256i vw_pr = _mm256_set1_epi64x(w_pr);

    __m256i vmod = _mm256_set1_epi64x(mod.n);

    __m256i cmp, mask;

    slong i;
    for (i=0; i+3<len; i+=4)
    {
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b+i));

        __m256i vq_hi = avx2_mulhi_split_v2(vw_pr, vb);

        __m256i llo, rlo;
        llo = avx2_mullo_epi64(vw, vb);
        rlo = avx2_mullo_epi64(vq_hi, vmod);
        __m256i vres = _mm256_sub_epi64(llo, rlo); // only low part is needed 

        // vres correct up to n or 2n
        cmp = _mm256_cmpgt_epi64(vmod, vres);
        mask = _mm256_andnot_si256(cmp, vmod);
        vres = _mm256_sub_epi64(vres, mask);

        cmp = _mm256_cmpgt_epi64(vmod, vres);
        mask = _mm256_andnot_si256(cmp, vmod);
        vres = _mm256_sub_epi64(vres, mask);

        _mm256_storeu_si256((__m256i *)(res+i), vres);
    }

    ulong p_hi=0, p_lo=0;
    ulong tmp;

    for ( ; i<len; i++)
    {
        umul_ppmm(p_hi, p_lo, b[i], w_pr);
        tmp = w*b[i] - p_hi*mod.n;
        
        if (tmp >= mod.n)
            tmp = tmp - mod.n;

        res[i] = tmp;
    }
}

void avx2_shoup_scalar_vector_mod_64_unrolled(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod)
{
    ulong w_pr = n_mulmod_precomp_shoup(w, mod.n);

    __m256i vw = _mm256_set1_epi64x(w);
    __m256i vw_pr = _mm256_set1_epi64x(w_pr);

    __m256i vmod = _mm256_set1_epi64x(mod.n);

    __m256i vb, vres;

    slong i;
    for (i = 0; i+31 < len; i += 32)
    {
        vb = _mm256_loadu_si256((const __m256i *)&b[i+0]);
        vres = _mm256_sub_epi64(avx2_mullo_epi64(vw, vb), avx2_mullo_epi64(avx2_mulhi_split_v2(vw_pr, vb), vmod)); // only low part is needed 
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        _mm256_storeu_si256((__m256i *)&res[i+0], vres);
        
        vb = _mm256_loadu_si256((const __m256i *)&b[i+4]);
        vres = _mm256_sub_epi64(avx2_mullo_epi64(vw, vb), avx2_mullo_epi64(avx2_mulhi_split_v2(vw_pr, vb), vmod)); 
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        _mm256_storeu_si256((__m256i *)&res[i+4], vres);
        
        vb = _mm256_loadu_si256((const __m256i *)&b[i+8]);
        vres = _mm256_sub_epi64(avx2_mullo_epi64(vw, vb), avx2_mullo_epi64(avx2_mulhi_split_v2(vw_pr, vb), vmod)); 
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        _mm256_storeu_si256((__m256i *)&res[i+8], vres);
        
        vb = _mm256_loadu_si256((const __m256i *)&b[i+12]);
        vres = _mm256_sub_epi64(avx2_mullo_epi64(vw, vb), avx2_mullo_epi64(avx2_mulhi_split_v2(vw_pr, vb), vmod)); 
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        _mm256_storeu_si256((__m256i *)&res[i+12], vres);

        vb = _mm256_loadu_si256((const __m256i *)&b[i+16]);
        vres = _mm256_sub_epi64(avx2_mullo_epi64(vw, vb), avx2_mullo_epi64(avx2_mulhi_split_v2(vw_pr, vb), vmod)); 
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        _mm256_storeu_si256((__m256i *)&res[i+16], vres);
        
        vb = _mm256_loadu_si256((const __m256i *)&b[i+20]);
        vres = _mm256_sub_epi64(avx2_mullo_epi64(vw, vb), avx2_mullo_epi64(avx2_mulhi_split_v2(vw_pr, vb), vmod)); 
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        _mm256_storeu_si256((__m256i *)&res[i+20], vres);

        vb = _mm256_loadu_si256((const __m256i *)&b[i+24]);
        vres = _mm256_sub_epi64(avx2_mullo_epi64(vw, vb), avx2_mullo_epi64(avx2_mulhi_split_v2(vw_pr, vb), vmod)); 
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        _mm256_storeu_si256((__m256i *)&res[i+24], vres);
        
        vb = _mm256_loadu_si256((const __m256i *)&b[i+28]);
        vres = _mm256_sub_epi64(avx2_mullo_epi64(vw, vb), avx2_mullo_epi64(avx2_mulhi_split_v2(vw_pr, vb), vmod)); 
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        _mm256_storeu_si256((__m256i *)&res[i+28], vres);
    }

    for ( ; i+3 < len; i+= 4)
    {
        vb = _mm256_loadu_si256((const __m256i *)&b[i]);
        vres = _mm256_sub_epi64(avx2_mullo_epi64(vw, vb), avx2_mullo_epi64(avx2_mulhi_split_v2(vw_pr, vb), vmod)); 
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        vres = _mm256_sub_epi64(vres, _mm256_andnot_si256(_mm256_cmpgt_epi64(vmod, vres), vmod));
        _mm256_storeu_si256((__m256i *)&res[i], vres);
    }

    ulong p_hi=0, p_lo=0;
    ulong tmp;

    for ( ; i<len; i++)
    {
        umul_ppmm(p_hi, p_lo, b[i], w_pr);
        tmp = w*b[i] - p_hi*mod.n;
        
        if (tmp >= mod.n)
            tmp = tmp - mod.n;

        res[i] = tmp;
    }
}

#if defined(__AVX512F__)
void avx512_shoup_scalar_vector_mod_64(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod)
{
    ulong w_pr = n_mulmod_precomp_shoup(w, mod.n);

    __m512i vw = _mm512_set1_epi64(w);
    __m512i vw_pr = _mm512_set1_epi64(w_pr);

    __m512i vmod = _mm512_set1_epi64(mod.n);

    slong i;
    for (i = 0; i+7 < len; i += 8)
    {
        __m512i vb = _mm512_loadu_si512((const __m512i *)(b+i));

        __m512i vq_hi = avx512_mulhi_split_v2(vw_pr, vb);

        __m512i llo, rlo;
        llo = _mm512_mullo_epi64(vw, vb);
        rlo = _mm512_mullo_epi64(vq_hi, vmod);
        __m512i vres = _mm512_sub_epi64(llo, rlo);

        vres = _mm512_min_epu64(_mm512_sub_epi64(vres, vmod), vres);
        vres = _mm512_min_epu64(_mm512_sub_epi64(vres, vmod), vres);

        _mm512_storeu_si512((__m512i *)(res+i), vres);
    }

    ulong p_hi=0, p_lo=0;
    ulong tmp;

    for ( ; i<len; i++)
    {
        umul_ppmm(p_hi, p_lo, b[i], w_pr);
        tmp = w*b[i] - p_hi*mod.n;
        
        if (tmp >= mod.n)
            tmp = tmp - mod.n;

        res[i] = tmp;
    }
}

void avx512_shoup_scalar_vector_mod_64_unrolled(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod)
{
    ulong w_pr = n_mulmod_precomp_shoup(w, mod.n);

    __m512i vw = _mm512_set1_epi64(w);
    __m512i vw_pr = _mm512_set1_epi64(w_pr);

    __m512i vmod = _mm512_set1_epi64(mod.n);

    __m512i vb, vres;

    slong i;
    for (i = 0; i+31 < len; i += 32)
    {
        vb = _mm512_loadu_si512((const __m512i *)&b[i+0]);
        vres = _mm512_sub_epi64(_mm512_mullo_epi64(vw, vb), _mm512_mullo_epi64(avx512_mulhi_split_v2(vw_pr, vb), vmod));
        vres = _mm512_min_epu64(_mm512_sub_epi64(vres, vmod), vres);
        vres = _mm512_min_epu64(_mm512_sub_epi64(vres, vmod), vres);
        _mm512_storeu_si512((__m512i *)&res[i+0], vres);
        vb = _mm512_loadu_si512((const __m512i *)&b[i+8]);
        vres = _mm512_sub_epi64(_mm512_mullo_epi64(vw, vb), _mm512_mullo_epi64(avx512_mulhi_split_v2(vw_pr, vb), vmod));
        vres = _mm512_min_epu64(_mm512_sub_epi64(vres, vmod), vres);
        vres = _mm512_min_epu64(_mm512_sub_epi64(vres, vmod), vres);
        _mm512_storeu_si512((__m512i *)&res[i+8], vres);
        vb = _mm512_loadu_si512((const __m512i *)&b[i+16]);
        vres = _mm512_sub_epi64(_mm512_mullo_epi64(vw, vb), _mm512_mullo_epi64(avx512_mulhi_split_v2(vw_pr, vb), vmod));
        vres = _mm512_min_epu64(_mm512_sub_epi64(vres, vmod), vres);
        vres = _mm512_min_epu64(_mm512_sub_epi64(vres, vmod), vres);
        _mm512_storeu_si512((__m512i *)&res[i+16], vres);
        vb = _mm512_loadu_si512((const __m512i *)&b[i+24]);
        vres = _mm512_sub_epi64(_mm512_mullo_epi64(vw, vb), _mm512_mullo_epi64(avx512_mulhi_split_v2(vw_pr, vb), vmod));
        vres = _mm512_min_epu64(_mm512_sub_epi64(vres, vmod), vres);
        vres = _mm512_min_epu64(_mm512_sub_epi64(vres, vmod), vres);
        _mm512_storeu_si512((__m512i *)&res[i+24], vres);
    }

    for ( ; i+7 < len; i += 8)
    {
        vb = _mm512_loadu_si512((const __m512i *)&b[i]);
        vres = _mm512_sub_epi64(_mm512_mullo_epi64(vw, vb), _mm512_mullo_epi64(avx512_mulhi_split_v2(vw_pr, vb), vmod));
        vres = _mm512_min_epu64(_mm512_sub_epi64(vres, vmod), vres);
        vres = _mm512_min_epu64(_mm512_sub_epi64(vres, vmod), vres);
        _mm512_storeu_si512((__m512i *)&res[i], vres);
    }

    ulong p_hi=0, p_lo=0;
    ulong tmp;

    for ( ; i<len; i++)
    {
        umul_ppmm(p_hi, p_lo, b[i], w_pr);
        tmp = w*b[i] - p_hi*mod.n;
        
        if (tmp >= mod.n)
            tmp = tmp - mod.n;

        res[i] = tmp;
    }
}
#endif