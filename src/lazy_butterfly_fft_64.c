#include "lazy_butterfly_fft_64.h"

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


static inline __m256i avx2_mulhi_split_lazy(__m256i a, __m256i b)
{
    __m256i r_hi, r_mi, r_lo;
    __m256i a_hi;
    __m256i b_hi;

    a_hi = _mm256_srli_epi64(a, 32);
    b_hi = _mm256_srli_epi64(b, 32);

    r_lo = _mm256_mul_epu32(a, b);
    r_hi = _mm256_mul_epu32(a_hi, b_hi);
    r_mi = _mm256_add_epi64(_mm256_mul_epu32(a, b_hi), _mm256_mul_epu32(a_hi, b));

    // detects the carry if any: https://stackoverflow.com/questions/32945410/sse2-intrinsics-comparing-unsigned-integers
    __m256i low = _mm256_add_epi64(r_lo, _mm256_slli_epi64(r_mi, 32));
    __m256i msb_mask = _mm256_set1_epi64x(1L << 63);
    __m256i xr_lo = _mm256_xor_si256(r_lo, msb_mask);
    __m256i xlow  = _mm256_xor_si256(low, msb_mask);
    __m256i carry = _mm256_cmpgt_epi64(xr_lo, xlow);
    carry = _mm256_srli_epi64(carry, 63);

    return _mm256_add_epi64(carry, _mm256_add_epi64(_mm256_srli_epi64(r_mi, 32), r_hi));
}

// improved variant, which should give the same thing
// (potentially missing carry, so only off by 1 at most?
// if yes, could we accept this in the FFT context?)
static inline __m256i avx2_mulhi_split_lazy_v2(__m256i a, __m256i b)
{
    // returns high part of the product of a and b over at most 64 bits integers
    // using avx2 intrinsics.

    __m256i r_hi, r_mi;
    __m256i a_hi;
    __m256i b_hi;

    a_hi = _mm256_srli_epi64(a, 32);
    b_hi = _mm256_srli_epi64(b, 32);

    r_mi = _mm256_add_epi64(_mm256_mul_epu32(a, b_hi), _mm256_mul_epu32(a_hi, b));
    r_hi = _mm256_mul_epu32(a_hi, b_hi);

    return _mm256_add_epi64(_mm256_srli_epi64(r_mi, 32), r_hi);
}

static inline __m256i avx2_mullo_split_lazy(__m256i a, __m256i b)
{
    // returns low part of the product of a and b over at most 64 bits integers
    // using avx2 intrinsics.

    __m256i r_mi, r_lo;
    __m256i a_hi;
    __m256i b_hi;

    a_hi = _mm256_srli_epi64(a, 32);
    b_hi = _mm256_srli_epi64(b, 32);

    r_lo = _mm256_mul_epu32(a, b);
    r_mi = _mm256_add_epi64(_mm256_mul_epu32(a, b_hi), _mm256_mul_epu32(a_hi, b));

    return _mm256_add_epi64(r_lo, _mm256_slli_epi64(r_mi, 32));
}

// https://stackoverflow.com/questions/37296289/fastest-way-to-multiply-an-array-of-int64-t/37320416#37320416
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

    slong i;
    for (i=0; i+3<len; i+=4)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a+i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b+i));

        // (a[i] < n2) ? a[i] : a[i] - n2
        __m256i cmp = _mm256_cmpgt_epi64(vmod2, va);
        __m256i mask = _mm256_andnot_si256(cmp, vmod2);
        va = _mm256_sub_epi64(va, mask);

        __m256i vq_hi = avx2_mulhi_split_lazy_v2(vw_pr, vb);

        __m256i llo, rlo;
        llo = avx2_mullo_epi64(vw, vb);
        rlo = avx2_mullo_epi64(vq_hi, vmod);
        __m256i vres = _mm256_sub_epi64(llo, rlo); // only low part is needed 

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
static inline __m512i avx512_mulhi_split_lazy(__m512i a, __m512i b)
{
    // returns high part of the product of a and b over at most 64 bits integers
    // using avx2 intrinsics.

    __m512i r_hi, r_mi, r_lo;
    __m512i a_hi;
    __m512i b_hi;

    a_hi = _mm512_srli_epi64(a, 32);
    b_hi = _mm512_srli_epi64(b, 32);

    r_lo = _mm512_mul_epu32(a, b);
    r_hi = _mm512_mul_epu32(a_hi, b_hi);
    r_mi = _mm512_add_epi64(_mm512_mul_epu32(a, b_hi), _mm512_mul_epu32(a_hi, b));

    // detects the carry if any
    __m512i low = _mm512_add_epi64(r_lo, _mm512_slli_epi64(r_mi, 32));
    __m512i msb_mask = _mm512_set1_epi64(1L << 63);
    __m512i xr_lo = _mm512_xor_si512(r_lo, msb_mask);
    __m512i xlow  = _mm512_xor_si512(low, msb_mask);
    __mmask8 cmask = _mm512_cmpgt_epi64_mask(xr_lo, xlow);
	__m512i carry = _mm512_maskz_set1_epi64(cmask, 1);

    return _mm512_add_epi64(carry, _mm512_add_epi64(_mm512_srli_epi64(r_mi, 32), r_hi));
}

static inline __m512i avx512_mulhi_split_lazy_v2(__m512i a, __m512i b)
//#define HAVE_AVX512_IFMA
#ifdef HAVE_AVX512_IFMA
{
    return _mm512_madd52hi_epu64(_mm512_setzero_si512(), a, b);
}
#else
{
    __m512i r_hi, r_mi;
    __m512i a_hi;
    __m512i b_hi;

    a_hi = _mm512_srli_epi64(a, 32);
    b_hi = _mm512_srli_epi64(b, 32);

    r_mi = _mm512_add_epi64(_mm512_mul_epu32(a, b_hi), _mm512_mul_epu32(a_hi, b));
    r_hi = _mm512_mul_epu32(a_hi, b_hi);

    return _mm512_add_epi64(_mm512_srli_epi64(r_mi, 32), r_hi);
}
#endif

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

        // (a[i] >= n2) ? a[i] - n2 : a[i]
        __mmask8 mask = _mm512_cmpge_epi64_mask(va, vmod2); // 1111111 if a[i] >= n2, 0 else
        __m512i tmp2 = _mm512_maskz_set1_epi64(mask, n2);
        va = _mm512_sub_epi64(va, tmp2);

        __m512i vq_hi = avx512_mulhi_split_lazy_v2(vw_pr, vb);

        __m512i llo, rlo;
        llo = _mm512_mullo_epi64(vw, vb);
        rlo = _mm512_mullo_epi64(vq_hi, vmod);
        __m512i vres = _mm512_sub_epi64(llo, rlo);

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
