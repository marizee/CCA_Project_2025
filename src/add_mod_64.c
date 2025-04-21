#include "add_mod_64.h"

__attribute__((optimize("-fno-tree-vectorize")))
void seq_add_mod(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    // Computes the modular addition of the vectors `a` and `b` and stores the result in `res`.
    // Requires `mod.n` <= 2**63 and coefficients already reduced mod n.

    for (slong i = 0; i < len; i++)
        res[i] = nmod_add(a[i], b[i], mod);
}

void seq_add_mod_vectorized(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    // Auto-vectorization of seq_add_mod.

    for (slong i = 0; i < len; i++)
        res[i] = nmod_add(a[i], b[i], mod);
}

void seq_add_mod_unrolled(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    // Loop-unrolling of seq_add_mod.

    slong i;
    for (i = 0; i+3 < len; i+=4)
    {
        res[i+0] = nmod_add(a[i+0], b[i+0], mod);
        res[i+1] = nmod_add(a[i+1], b[i+1], mod);
        res[i+2] = nmod_add(a[i+2], b[i+2], mod);
        res[i+3] = nmod_add(a[i+3], b[i+3], mod);
    }

    // when len is not a multiple of 4
    for ( ; i < len; i++)
        res[i] = nmod_add(a[i], b[i], mod);
}

void simd2_add_mod(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    // Computes the modular addition of the vectors `a` and `b` and stores the result in `res` using avx2.
    // Requires `mod.n` <= 2**63 and coefficients already reduced mod n.

    __m256i vmod = _mm256_set1_epi64x(mod.n);
    __m256i vzero = _mm256_setzero_si256();

    slong i;
    for (i=0; i+3 < len; i+=4)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)&a[i]);
        __m256i vb = _mm256_loadu_si256((const __m256i *)&b[i]);
        __m256i add = _mm256_add_epi64(va, vb);

        // modular reduction
        add = _mm256_sub_epi64(add, vmod);
        __m256i cmp = _mm256_cmpgt_epi64(vzero, add);
        __m256i mask = _mm256_and_si256(cmp, vmod);
        add = _mm256_add_epi64(add, mask);

        _mm256_storeu_si256((__m256i *)&res[i], add);
    }

    // if len is not a multiple of 4
    for ( ; i < len; i++)
        res[i] = nmod_add(a[i], b[i], mod);
}

void simd2_add_mod_unrolled(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    // Loop-unrolling of simd2_add_mod.

    __m256i add;
    __m256i vmod = _mm256_set1_epi64x(mod.n);
    __m256i vzero = _mm256_setzero_si256();

    slong i;
    for (i=0; i+31 < len; i+=32)
    {
        add = _mm256_sub_epi64(_mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+ 0]), _mm256_loadu_si256((const __m256i *)&b[i+ 0])),vmod);
        _mm256_storeu_si256((__m256i *)&res[i+ 0], _mm256_add_epi64(add, _mm256_and_si256(_mm256_cmpgt_epi64(vzero, add), vmod)));
        add = _mm256_sub_epi64(_mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+ 4]), _mm256_loadu_si256((const __m256i *)&b[i+ 4])),vmod);
        _mm256_storeu_si256((__m256i *)&res[i+ 4], _mm256_add_epi64(add, _mm256_and_si256(_mm256_cmpgt_epi64(vzero, add), vmod)));
        add = _mm256_sub_epi64(_mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+ 8]), _mm256_loadu_si256((const __m256i *)&b[i+ 8])),vmod);
        _mm256_storeu_si256((__m256i *)&res[i+ 8], _mm256_add_epi64(add, _mm256_and_si256(_mm256_cmpgt_epi64(vzero, add), vmod)));
        add = _mm256_sub_epi64(_mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+12]), _mm256_loadu_si256((const __m256i *)&b[i+12])),vmod);
        _mm256_storeu_si256((__m256i *)&res[i+12], _mm256_add_epi64(add, _mm256_and_si256(_mm256_cmpgt_epi64(vzero, add), vmod)));
        add = _mm256_sub_epi64(_mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+16]), _mm256_loadu_si256((const __m256i *)&b[i+16])),vmod);
        _mm256_storeu_si256((__m256i *)&res[i+16], _mm256_add_epi64(add, _mm256_and_si256(_mm256_cmpgt_epi64(vzero, add), vmod)));
        add = _mm256_sub_epi64(_mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+20]), _mm256_loadu_si256((const __m256i *)&b[i+20])),vmod);
        _mm256_storeu_si256((__m256i *)&res[i+20], _mm256_add_epi64(add, _mm256_and_si256(_mm256_cmpgt_epi64(vzero, add), vmod)));
        add = _mm256_sub_epi64(_mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+24]), _mm256_loadu_si256((const __m256i *)&b[i+24])),vmod);
        _mm256_storeu_si256((__m256i *)&res[i+24], _mm256_add_epi64(add, _mm256_and_si256(_mm256_cmpgt_epi64(vzero, add), vmod)));
        add = _mm256_sub_epi64(_mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+28]), _mm256_loadu_si256((const __m256i *)&b[i+28])),vmod);
        _mm256_storeu_si256((__m256i *)&res[i+28], _mm256_add_epi64(add, _mm256_and_si256(_mm256_cmpgt_epi64(vzero, add), vmod)));
    }
    
    for ( ; i+4 < len; i+=4)
    {
        add = _mm256_sub_epi64(_mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i]), _mm256_loadu_si256((const __m256i *)&b[i])),vmod);
        _mm256_storeu_si256((__m256i *)&res[i], _mm256_add_epi64(add, _mm256_and_si256(_mm256_cmpgt_epi64(vzero, add), vmod)));
    }
    
    for ( ; i < len; i++)
        res[i] = nmod_add(a[i], b[i], mod);
}

#if defined(__AVX512F__)
void simd512_add_mod(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    // Computes the modular addition of the vectors `a` and `b` and stores the result in `res` using avx512.
    // Requires `mod.n` <= 2**63 and coefficients already reduced mod n.

    __m512i vmod = _mm512_set1_epi64(mod.n);
    __m512i vzero = _mm512_setzero_si512();

    slong i;
    for (i=0; i+7 < len; i+=8)
    {
        __m512i va = _mm512_loadu_si512((const __m512i *)&a[i]);
        __m512i vb = _mm512_loadu_si512((const __m512i *)&b[i]);
        __m512i add = _mm512_add_epi64(va, vb);

        // modular reduction
        add = _mm512_sub_epi64(add, vmod);
        __mmask8 cmp = _mm512_cmpgt_epi64_mask (vzero, add);
        __m512i mask = _mm512_maskz_set1_epi64(cmp, vmod);
        add = _mm512_add_epi64(add, mask);

        _mm512_storeu_si512((__m512i *)&res[i], add);
    }

    // if len is not a multiple of 8
    for ( ; i < len; i++)
        res[i] = nmod_add(a[i], b[i], mod);
}

void simd512_add_mod_unrolled(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    // Loop-unrolling of simd512_add_mod.

    __m512i vmod = _mm512_set1_epi64(mod.n);
    __m512i vzero = _mm512_setzero_si512();

    slong i;
    for (i=0; i+31 < len; i+=32)
    {
        add = _mm512_sub_epi64(_mm512_add_epi64(_mm512_loadu_si512((const __m512i *)&a[i+ 0]), _mm512_loadu_si512((const __m512i *)&b[i+ 0])), vmod);
        add = _mm512_sub_epi64(_mm512_add_epi64(_mm512_loadu_si512((const __m512i *)&a[i+ 8]), _mm512_loadu_si512((const __m512i *)&b[i+ 8])), vmod);
        add = _mm512_sub_epi64(_mm512_add_epi64(_mm512_loadu_si512((const __m512i *)&a[i+16]), _mm512_loadu_si512((const __m512i *)&b[i+16])), vmod);
        add = _mm512_sub_epi64(_mm512_add_epi64(_mm512_loadu_si512((const __m512i *)&a[i+24]), _mm512_loadu_si512((const __m512i *)&b[i+24])), vmod);

        _mm512_storeu_si512((__m512i *)&res[i+ 0], _mm512_add_epi64(add, _mm512_maskz_set1_epi64(_mm512_cmpgt_epi64_mask (vzero, add), vmod)));
        _mm512_storeu_si512((__m512i *)&res[i+ 8], _mm512_add_epi64(add, _mm512_maskz_set1_epi64(_mm512_cmpgt_epi64_mask (vzero, add), vmod)));
        _mm512_storeu_si512((__m512i *)&res[i+16], _mm512_add_epi64(add, _mm512_maskz_set1_epi64(_mm512_cmpgt_epi64_mask (vzero, add), vmod)));
        _mm512_storeu_si512((__m512i *)&res[i+24], _mm512_add_epi64(add, _mm512_maskz_set1_epi64(_mm512_cmpgt_epi64_mask (vzero, add), vmod)));
    }

    for ( ; i+7 < len; i+=8)
    {
        add = _mm256_sub_epi64(_mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i]), _mm256_loadu_si256((const __m256i *)&b[i])),vmod);
        _mm256_storeu_si256((__m256i *)&res[i], _mm256_add_epi64(add, _mm256_and_si256(_mm256_cmpgt_epi64(vzero, add), vmod)));
    }

    // if len is not a multiple of 8
    for ( ; i < len; i++)
        res[i] = nmod_add(a[i], b[i], mod);
}
#endif
