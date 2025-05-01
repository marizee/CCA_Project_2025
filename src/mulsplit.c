#include "mulsplit.h"

#define MASK ((1L << 32) - 1)

// HIGH PART
//      a,b < 2**62
//      computes 64 bits high part of the product between a and b with split at 32
//      compared:
//      - umul_ppmm
//      - naive
//      - avx2
//      - avx512 mine

ulong mulhi_split(ulong a, ulong b)
{
    ulong r_hi, r_mi, r_lo;
    ulong a_lo, a_hi;
    ulong b_lo, b_hi;

    a_lo = a & MASK;
    a_hi = a >> 32;
    b_lo = b & MASK;
    b_hi = b >> 32;

    r_lo = a_lo*b_lo;
    r_hi = a_hi*b_hi;
    r_mi = a_lo*b_hi + a_hi*b_lo;

    // detects the carry if any
    ulong low = r_lo + (r_mi << 32);
    ulong carry = (low < r_lo ? 1 : 0);

    return (r_mi >> 32) + r_hi + carry;
}

__m256i avx2_mulhi_split(__m256i a, __m256i b)
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

#if defined(__AVX512F__)
__m512i avx512_mulhi_split(__m512i a, __m512i b)
{
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
#endif

// LOW PART
//      a,b < 2**62
//      computes low part of the product between a and b with split at 32
//      - umul_ppmm
//      - naive
//      - avx2 mine
//      - avx2 stack
//      - avx512 mullo

ulong mullo_split(ulong a, ulong b)
{
    ulong r_mi, r_lo;
    ulong a_lo, a_hi;
    ulong b_lo, b_hi;

    a_lo = a & MASK;
    a_hi = a >> 32;
    b_lo = b & MASK;
    b_hi = b >> 32;

    r_lo = a_lo*b_lo;
    r_mi = a_lo*b_hi + a_hi*b_lo;

    return r_lo + (r_mi << 32);
}

__m256i avx2_mullo_split(__m256i a, __m256i b)
{
    __m256i r_mi, r_lo;
    __m256i a_hi;
    __m256i b_hi;

    a_hi = _mm256_srli_epi64(a, 32);
    b_hi = _mm256_srli_epi64(b, 32);

    r_lo = _mm256_mul_epu32(a, b);
    r_mi = _mm256_add_epi64(_mm256_mul_epu32(a, b_hi), _mm256_mul_epu32(a_hi, b));

    return _mm256_add_epi64(r_lo, _mm256_slli_epi64(r_mi, 32));
}

__m256i avx2_mullo_epi64(__m256i a, __m256i b)
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

// BOTH
void avx2_mul_split(__m256i* hi, __m256i* lo, __m256i a, __m256i b)
{
    __m256i r_hi, r_mi, r_lo;
    __m256i a_hi;
    __m256i b_hi;

    a_hi = _mm256_srli_epi64(a, 32);
    b_hi = _mm256_srli_epi64(b, 32);

    r_lo = _mm256_mul_epu32(a, b);
    r_hi = _mm256_mul_epu32(a_hi, b_hi);
    r_mi = _mm256_add_epi64(_mm256_mul_epu32(a, b_hi), _mm256_mul_epu32(a_hi, b));

    // detects carry for high part
    __m256i low = _mm256_add_epi64(r_lo, _mm256_slli_epi64(r_mi, 32));
    __m256i msb_mask = _mm256_set1_epi64x(1L << 63);
    __m256i xr_lo = _mm256_xor_si256(r_lo, msb_mask);
    __m256i xlow  = _mm256_xor_si256(low, msb_mask);
    __m256i carry = _mm256_cmpgt_epi64(xr_lo, xlow);
    carry = _mm256_srli_epi64(carry, 63);

    *lo = _mm256_add_epi64(r_lo, _mm256_slli_epi64(r_mi, 32));
    *hi = _mm256_add_epi64(carry, _mm256_add_epi64(_mm256_srli_epi64(r_mi, 32), r_hi));
}

//__m512i avx512_mul_split(__m512i hi, __m512i lo, __m512i a, __m512i b) ;