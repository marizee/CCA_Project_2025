#include "add_64.h"

__attribute__((optimize("-fno-tree-vectorize")))
void seq_add(nn_ptr res, nn_ptr a, nn_ptr b, slong len)
{
    // Computes the addition of the vectors `a` and `b` and stores the result in `res`.
    // Requires coefficients < 2**63.

    for (slong i = 0; i < len; i++)
        res[i] = a[i] + b[i];
}

void seq_add_vectorized(nn_ptr res, nn_ptr a, nn_ptr b, slong len)
{
    // Auto-vectorization of seq_add.

    for (slong i = 0; i < len; i++)
        res[i] = a[i] + b[i];
}

void seq_add_unrolled(nn_ptr res, nn_ptr a, nn_ptr b, slong len)
{
    // Loop-unrolling of seq_add.

    slong i;
    for (i = 0; i+3 < len; i+=4)
    {
        res[i+0] = a[i+0] + b[i+0];
        res[i+1] = a[i+1] + b[i+1];
        res[i+2] = a[i+2] + b[i+2];
        res[i+3] = a[i+3] + b[i+3];
    }

    // when len is not a multiple of 4
    for ( ; i < len; i++)
        res[i] = a[i] + b[i];
}

void simd2_add(nn_ptr res, nn_ptr a, nn_ptr b, slong len)
{
    // Computes the addition of the vectors `a` and `b` and stores the result in `res` using avx2.
    // Requires coefficients < 2**63.

    slong i;
    for (i=0; i+3 < len; i+=4)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)&a[i]);
        __m256i vb = _mm256_loadu_si256((const __m256i *)&b[i]);
        __m256i add = _mm256_add_epi64(va, vb);
        _mm256_storeu_si256((__m256i *)&res[i], add);
    }

    // if len is not a multiple of 4
    for ( ; i < len; i++)
        res[i] = a[i] + b[i];
}

void simd2_add_unrolled(nn_ptr res, nn_ptr a, nn_ptr b, slong len)
{
    // Loop-unrolling of simd2_add.

    slong i;    
    for (i=0; i+31 < len; i+=32)
    {
        _mm256_storeu_si256((__m256i *)&res[i+ 0], _mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+ 0]), _mm256_loadu_si256((const __m256i *)&b[i+ 0])));
        _mm256_storeu_si256((__m256i *)&res[i+ 4], _mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+ 4]), _mm256_loadu_si256((const __m256i *)&b[i+ 4])));
        _mm256_storeu_si256((__m256i *)&res[i+ 8], _mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+ 8]), _mm256_loadu_si256((const __m256i *)&b[i+ 8])));
        _mm256_storeu_si256((__m256i *)&res[i+12], _mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+12]), _mm256_loadu_si256((const __m256i *)&b[i+12])));
        _mm256_storeu_si256((__m256i *)&res[i+16], _mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+16]), _mm256_loadu_si256((const __m256i *)&b[i+16])));
        _mm256_storeu_si256((__m256i *)&res[i+20], _mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+20]), _mm256_loadu_si256((const __m256i *)&b[i+20])));
        _mm256_storeu_si256((__m256i *)&res[i+24], _mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+24]), _mm256_loadu_si256((const __m256i *)&b[i+24])));
        _mm256_storeu_si256((__m256i *)&res[i+28], _mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i+28]), _mm256_loadu_si256((const __m256i *)&b[i+28])));
    }
    
    for ( ; i+4 < len; i+=4)
        _mm256_storeu_si256((__m256i *)&res[i], _mm256_add_epi64(_mm256_loadu_si256((const __m256i *)&a[i]), _mm256_loadu_si256((const __m256i *)&b[i])));

    for ( ; i < len; i++)
        res[i] = a[i] + b[i];
}

#if defined(__AVX512F__)
void simd512_add(nn_ptr res, nn_ptr a, nn_ptr b, slong len)
{
    // Computes the addition of the vectors `a` and `b` and stores the result in `res` using avx512.
    // Requires coefficients < 2**63.

    slong i;
    for (i=0; i+7 < len; i+=8)
    {
        __m512i va = _mm512_loadu_si512((const __m512i *)&a[i]);
        __m512i vb = _mm512_loadu_si512((const __m512i *)&b[i]);
        __m512i add = _mm512_add_epi64(va, vb);
        _mm512_storeu_si512((__m512i *)&res[i], add);
    }

    // if len is not a multiple of 8
    for ( ; i < len; i++)
        res[i] = a[i] + b[i];
}

void simd512_add_unrolled(nn_ptr res, nn_ptr a, nn_ptr b, slong len)
{
    // Loop-unrolling of simd512_add.

    slong i;
    for (i=0; i+31 < len; i+=32)
    {
        _mm512_storeu_si512((__m512i *)&res[i+ 0], _mm512_add_epi64(_mm512_loadu_si512((const __m512i *)&a[i+ 0]), _mm512_loadu_si512((const __m512i *)&b[i+ 0])));
        _mm512_storeu_si512((__m512i *)&res[i+ 8], _mm512_add_epi64(_mm512_loadu_si512((const __m512i *)&a[i+ 8]), _mm512_loadu_si512((const __m512i *)&b[i+ 8])));
        _mm512_storeu_si512((__m512i *)&res[i+16], _mm512_add_epi64(_mm512_loadu_si512((const __m512i *)&a[i+16]), _mm512_loadu_si512((const __m512i *)&b[i+16])));
        _mm512_storeu_si512((__m512i *)&res[i+24], _mm512_add_epi64(_mm512_loadu_si512((const __m512i *)&a[i+24]), _mm512_loadu_si512((const __m512i *)&b[i+24])));
    }

    for ( ; i+7 < len; i+=8)
        _mm512_storeu_si512((__m512i *)&res[i], _mm512_add_epi64(_mm512_loadu_si512((const __m512i *)&a[i]), _mm512_loadu_si512((const __m512i *)&b[i])));

    for ( ; i < len; i++)
        res[i] = a[i] + b[i];
}
#endif
