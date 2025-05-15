#include "lazy_butterfly_fft_64.h"
#include "mulsplit.h"

void preinv_fft_lazy44(nn_ptr a, nn_ptr b, ulong w, ulong w_pr, slong len, ulong n, ulong n2, ulong p_hi, ulong p_lo, ulong tmp)
{
    // requirements:
    //      - n < 2**62
    //      - w < n
    //      - coeffs of a, b < 4*n
    for (slong i=0; i<len; i++)
    {
        ulong u = a[i];
        ulong v = b[i];

        if (u >= n2)
            u -= n2;

        umul_ppmm(p_hi, p_lo, v, w_pr);
        tmp = w*v - p_hi*n;
        a[i] = u + tmp;
        b[i] = u - tmp + n2;
    }
}

void avx2_preinv_split_fft_lazy44(nn_ptr a, nn_ptr b, ulong w, ulong w_pr, slong len, ulong n, ulong n2, ulong p_hi, ulong p_lo, ulong tmp)
{
    // requirements:
    //      - n < 2**62
    //      - w < n
    //      - coeffs of a, b < 4*n

    __m256i vw = _mm256_set1_epi64x(w);
    __m256i vw_pr = _mm256_set1_epi64x(w_pr);

    __m256i vmod = _mm256_set1_epi64x(n);
    __m256i vmod2 = _mm256_set1_epi64x(n2);

    __m256i cmp, mask;

    slong i;
    for (i=0; i+3<len; i+=4)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a+i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b+i));

        // (a[i] < n2) ? a[i] : a[i] - n2
        cmp = _mm256_cmpgt_epi64(vmod2, va);
        mask = _mm256_andnot_si256(cmp, vmod2);
        va = _mm256_sub_epi64(va, mask);

        __m256i vq_hi = avx2_mulhi_split_v2(vw_pr, vb);

        __m256i llo, rlo;
        llo = avx2_mullo_epi64(vw, vb);
        rlo = avx2_mullo_epi64(vq_hi, vmod);
        __m256i vres = _mm256_sub_epi64(llo, rlo); // only low part is needed

        // vres either ok or +n or +2n => needs 1 mod red
        cmp = _mm256_cmpgt_epi64(vmod2, vres);
        mask = _mm256_andnot_si256(cmp, vmod2);
        vres = _mm256_sub_epi64(vres, mask);

        __m256i add = _mm256_add_epi64(va, vres);
        __m256i sub = _mm256_add_epi64(_mm256_sub_epi64(va, vres), vmod2);

        _mm256_storeu_si256((__m256i *)(a+i), add);
        _mm256_storeu_si256((__m256i *)(b+i), sub);
    }

    for ( ; i<len; i++)
    {
        ulong u = a[i];
        ulong v = b[i];

        if (u >= n2)
            u -= n2;

        umul_ppmm(p_hi, p_lo, v, w_pr);
        tmp = w*v - p_hi*n;
        a[i] = u + tmp;
        b[i] = u - tmp + n2;
    }
}

#if defined(__AVX512F__)
void avx512_preinv_split_fft_lazy44(nn_ptr a, nn_ptr b, ulong w, ulong w_pr, slong len, ulong n, ulong n2, ulong p_hi, ulong p_lo, ulong tmp)
{
    // requirements:
    //      - n < 2**62
    //      - w < n
    //      - coeffs of a, b < 4*n

    __m512i vw = _mm512_set1_epi64(w);
    __m512i vw_pr = _mm512_set1_epi64(w_pr);

    __m512i vmod = _mm512_set1_epi64(n);
    __m512i vmod2 = _mm512_set1_epi64(n2);

    slong i;
    for (i=0; i+7<len; i+=8)
    {
        __m512i va = _mm512_loadu_si512((const __m512i *)(a+i));
        __m512i vb = _mm512_loadu_si512((const __m512i *)(b+i));

        // (a[i] >= n2) ? a[i] - n2 : a[i] <=> min(a[i] - 2n, a[i])
	    va = _mm512_min_epu64(_mm512_sub_epi64(va, vmod2), va);

        __m512i vq_hi = avx512_mulhi_split_v2(vw_pr, vb);

        __m512i llo, rlo;
        llo = _mm512_mullo_epi64(vw, vb);
        rlo = _mm512_mullo_epi64(vq_hi, vmod);
        __m512i vres = _mm512_sub_epi64(llo, rlo);

	    vres = _mm512_min_epu64(_mm512_sub_epi64(vres, vmod2), vres);

        __m512i add = _mm512_add_epi64(va, vres);
        __m512i sub = _mm512_add_epi64(_mm512_sub_epi64(va, vres), vmod2);

        _mm512_storeu_si512((__m512i *)(a+i), add);
        _mm512_storeu_si512((__m512i *)(b+i), sub);
    }

    for ( ; i<len; i++)
    {
        ulong u = a[i];
        ulong v = b[i];

        if (u >= n2)
            u -= n2;

        umul_ppmm(p_hi, p_lo, v, w_pr);
        tmp = w*v - p_hi*n;
        a[i] = u + tmp;
        b[i] = u - tmp + n2;
    }
}

#endif
