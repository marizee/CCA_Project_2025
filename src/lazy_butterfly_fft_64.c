#include "lazy_butterfly_fft_64.h"
#include "mulsplit.h"

#define SPLIT 32
#define MASK ((1L << SPLIT) - 1)


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

/*
void avx2_mulhi_split_lazy(__m256i* high, __m256i a, __m256i b)
{
    // returns high part of the product of a and b over at most 64 bits integers
    // using avx2 intrinsics.

    __m256i r_hi, r_mi; //, r_lo;
    __m256i a_lo, a_hi;
    __m256i b_lo, b_hi;

    const __m256i vMASK = _mm256_set1_epi64x(MASK);

    a_lo = _mm256_and_si256(a, vMASK);
    a_hi = _mm256_srli_epi64(a, SPLIT);
    b_lo = _mm256_and_si256(b, vMASK);
    b_hi = _mm256_srli_epi64(b, SPLIT);

    //r_lo = _mm256_mul_epu32(a_lo, b_lo);
    r_hi = _mm256_mul_epu32(a_hi, b_hi);
    r_mi = _mm256_add_epi64(_mm256_mul_epu32(a_lo, b_hi), _mm256_mul_epu32(a_hi, b_lo));
    // FIXME explain why no overflow above

    // hi = (umi >> 38) + (uhi >> 12)
    *high = _mm256_add_epi64(_mm256_srli_epi64(r_mi, (64-SPLIT)), _mm256_srli_epi64(r_hi, (64-2*SPLIT)));
}

void avx2_mullo_split_lazy(__m256i* low, __m256i a, __m256i b)
{
    // returns low part of the product of a and b over at most 64 bits integers
    // using avx2 intrinsics.

    __m256i r_hi, r_mi, r_lo;
    __m256i a_lo, a_hi;
    __m256i b_lo, b_hi;
    const __m256i vMASK = _mm256_set1_epi64x(MASK);

    a_lo = _mm256_and_si256(a, vMASK);
    a_hi = _mm256_srli_epi64(a, SPLIT);
    b_lo = _mm256_and_si256(b, vMASK);
    b_hi = _mm256_srli_epi64(b, SPLIT);

    r_lo = _mm256_mul_epu32(a_lo, b_lo);
    r_hi = _mm256_mul_epu32(a_hi, b_hi);
    r_mi = _mm256_add_epi64(_mm256_mul_epu32(a_lo, b_hi), _mm256_mul_epu32(a_hi, b_lo));

    *low = _mm256_add_epi64(r_lo, _mm256_add_epi64(_mm256_slli_epi64(r_mi, SPLIT), _mm256_slli_epi64(r_hi, 2*SPLIT)));
}


static inline __m256i avx2_mullo_epi64(__m256i a, __m256i b)
{
    // There is no vpmullq until AVX-512. Split into 32-bit multiplies
    // Given a and b composed of high<<32 | low  32-bit halves
    // a*b = a_low*(u64)b_low  + (u64)(a_high*b_low + a_low*b_high)<<32;  // same for signed or unsigned a,b since we aren't widening to 128
    // the a_high * b_high product isn't needed for non-widening; its place value is entirely outside the low 64 bits.

    __m256i b_swap  = _mm256_shuffle_epi32(b, _MM_SHUFFLE(2,3, 0,1));   // swap H<->L
    __m256i crossprod  = _mm256_mullo_epi32(a, b_swap);                 // 32-bit L*H and H*L cross-products

    __m256i prodlh = _mm256_slli_epi64(crossprod, 32);          // bring the low half up to the top of each 64-bit chunk 
    __m256i prodhl = _mm256_and_si256(crossprod, _mm256_set1_epi64x(0xFFFFFFFF00000000)); // isolate the other, also into the high half were it needs to eventually be
    __m256i sumcross = _mm256_add_epi32(prodlh, prodhl);       // the sum of the cross products, with the low half of each u64 being 0.

    __m256i prodll  = _mm256_mul_epu32(a,b);                  // widening 32x32 => 64-bit  low x low products
    __m256i prod    = _mm256_add_epi32(prodll, sumcross);     // add the cross products into the high half of the result
    return  prod;
}
*/

void avx2_preinv_split_fft_lazy44(nn_ptr a, nn_ptr b, ulong w, ulong w_pr, slong len, ulong n, ulong n2, ulong p_hi, ulong p_lo, ulong tmp)
{
    // requirements:
    //      - n < 2**62
    //      - w < n
    //      - coeffs of a, b < 4*n

    __m256i vw = _mm256_set1_epi64x(w);
    __m256i vw_pr = _mm256_set1_epi64x(w_pr);

    __m256i vq_hi = _mm256_setzero_si256();
    __m256i vres = _mm256_setzero_si256();

    __m256i vmod = _mm256_set1_epi64x(n);
    __m256i vmod2 = _mm256_set1_epi64x(n2);

    slong i;
    for (i=0; i+3<len; i+=4)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a+i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b+i));

        // (a[i] < n2) ? a[i] : a[i] - n2
        __m256i cmp = _mm256_cmpgt_epi64(vmod2, va);
        __m256i mask = _mm256_andnot_si256(cmp, vmod2);
        va = _mm256_sub_epi64(va, mask);

        vq_hi = avx2_mulhi_split(vw_pr, vb);

        __m256i llo = avx2_mullo_split(vw, vb);
        __m256i rlo = avx2_mullo_split(vq_hi, vmod);
        vres = _mm256_sub_epi64(llo, rlo); // only low part is needed

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
/*
void avx512_mulhi_split_lazy(__m512i* high, __m512i a, __m512i b)
{
    // returns high part of the product of a and b over at most 64 bits integers
    // using avx2 intrinsics.

    __m512i r_hi, r_mi; //, r_lo;
    __m512i a_lo, a_hi;
    __m512i b_lo, b_hi;

    const __m512i vMASK = _mm512_set1_epi64(MASK);

    a_lo = _mm512_and_si512(a, vMASK);
    a_hi = _mm512_srli_epi64(a, SPLIT);
    b_lo = _mm512_and_si512(b, vMASK);
    b_hi = _mm512_srli_epi64(b, SPLIT);

    //r_lo = _mm256_mul_epu32(a_lo, b_lo);
    r_hi = _mm512_mul_epu32(a_hi, b_hi);
    r_mi = _mm512_add_epi64(_mm512_mul_epu32(a_lo, b_hi), _mm512_mul_epu32(a_hi, b_lo));

    // hi = (umi >> 38) + (uhi >> 12)
    *high = _mm512_add_epi64(_mm512_srli_epi64(r_mi, (64-SPLIT)), _mm512_srli_epi64(r_hi, (64-2*SPLIT)));
}
*/

void avx512_preinv_split_fft_lazy44(nn_ptr a, nn_ptr b, ulong w, ulong w_pr, slong len, ulong n, ulong n2, ulong p_hi, ulong p_lo, ulong tmp)
{
    // requirements:
    //      - n < 2**62
    //      - w < n
    //      - coeffs of a, b < 4*n

    __m512i vw = _mm512_set1_epi64(w);
    __m512i vw_pr = _mm512_set1_epi64(w_pr);

    __m512i vq_hi = _mm512_setzero_si512();
    __m512i vres = _mm512_setzero_si512();

    __m512i vmod = _mm512_set1_epi64(n);
    __m512i vmod2 = _mm512_set1_epi64(n2);

    slong i;
    for (i=0; i+7<len; i+=8)
    {
        __m512i va = _mm512_loadu_si512((const __m512i *)(a+i));
        __m512i vb = _mm512_loadu_si512((const __m512i *)(b+i));

        // (a[i] >= n2) ? a[i] - n2 : a[i]
        __mmask8 mask = _mm512_cmpge_epi64_mask(va, vmod2); // 1111111 if a[i] >= n2, 0 else
        __m512i tmp2 = _mm512_maskz_set1_epi64(mask, n2);
        va = _mm512_sub_epi64(va, tmp2);

        vq_hi = avx512_mulhi_split(vw_pr, vb);

        __m512i llo = _mm512_mullo_epi64(vw, vb);
        __m512i rlo = _mm512_mullo_epi64(vq_hi, vmod);
        vres = _mm512_sub_epi64(llo, rlo);

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
