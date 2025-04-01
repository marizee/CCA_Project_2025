#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"
#include "flint/profiler.h"


#define MASK ((1L << 32) - 1)

// HIGH PART
//      a,b < 2**62
//      computes high part of the product between a and b with split at 32
//      compared:
//      - umul_ppmm
//      - naive
//      - avx2

void mulhi_split(ulong* high, ulong a, ulong b)
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

    ulong low = r_lo + (r_mi << 32) + (r_hi << 64);
    ulong carry = (low < r_lo ? 1 : 0);

    *high = (r_lo >> 64) + (r_mi >> 32) + r_hi + carry;
}

void avx2_mulhi_split(nn_ptr high, nn_ptr a, nn_ptr b)
{
    __m256i va = _mm256_loadu_si256((const __m256i *)a);
    __m256i vb = _mm256_loadu_si256((const __m256i *)b);

    __m256i r_hi, r_mi, r_lo;
    __m256i a_lo, a_hi;
    __m256i b_lo, b_hi;
    const __m256i vMASK = _mm256_set1_epi64x(MASK);

    a_lo = _mm256_and_si256(va, vMASK);
    a_hi = _mm256_srli_epi64(va, 32);
    b_lo = _mm256_and_si256(vb, vMASK);
    b_hi = _mm256_srli_epi64(vb, 32);

    r_lo = _mm256_mul_epu32(a_lo, b_lo);
    r_hi = _mm256_mul_epu32(a_hi, b_hi);
    r_mi = _mm256_add_epi64(_mm256_mul_epu32(a_lo, b_hi), _mm256_mul_epu32(a_hi, b_lo));

    __m256i low = _mm256_add_epi64(r_lo, _mm256_slli_epi64(r_mi, 32));
    __m256i and = _mm256_and_si256(r_lo, low); // works
    __m256i carry = _mm256_cmpgt_epi64(and, r_lo);
    carry = _mm256_srli_epi64(carry, 63);

    __m256i tmp = _mm256_add_epi64(carry, _mm256_add_epi64(_mm256_srli_epi64(r_mi, 32), r_hi));
    _mm256_storeu_si256((__m256i *)high, tmp);
}

// LOW PART
//      a,b < 2**62
//      computes low part of the product between a and b with split at 32
//      - umul_ppmm
//      - naive
//      - avx2 (mine)

void mullo_split(ulong* low, ulong a, ulong b)
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

    *low = r_lo + (r_mi << 32);
}

void avx2_mullo_split(nn_ptr low, nn_ptr a, nn_ptr b)
{
    __m256i va = _mm256_loadu_si256((const __m256i *)a);
    __m256i vb = _mm256_loadu_si256((const __m256i *)b);

    __m256i r_mi, r_lo;
    __m256i a_lo, a_hi;
    __m256i b_lo, b_hi;
    const __m256i vMASK = _mm256_set1_epi64x(MASK);

    a_lo = _mm256_and_si256(va, vMASK);
    a_hi = _mm256_srli_epi64(va, 32);
    b_lo = _mm256_and_si256(vb, vMASK);
    b_hi = _mm256_srli_epi64(vb, 32);

    r_lo = _mm256_mul_epu32(a_lo, b_lo);
    r_mi = _mm256_add_epi64(_mm256_mul_epu32(a_lo, b_hi), _mm256_mul_epu32(a_hi, b_lo));

    __m256i tmp = _mm256_add_epi64(r_lo, _mm256_slli_epi64(r_mi, 32));
    _mm256_storeu_si256((__m256i *)low, tmp);

}

int main()
{
    FLINT_TEST_INIT(state);

    nn_ptr a, b;
    nn_ptr p_lo1, p_hi1;
    nn_ptr p_lo2, p_hi2;
    nn_ptr p_lo3, p_hi3;
    int l1, l2;
    int h1, h2;

    // init vector
    a = _nmod_vec_init(4); b = _nmod_vec_init(4);
    p_lo1 = _nmod_vec_init(4); p_hi1 = _nmod_vec_init(4);
    p_lo2 = _nmod_vec_init(4); p_hi2 = _nmod_vec_init(4);
    p_lo3 = _nmod_vec_init(4); p_hi3 = _nmod_vec_init(4);


    for (slong k=31; k <= 34; k++)
    {
        for (slong i = 0; i < 4; i++) 
        {
            a[i] = n_randbits(state, k);
            b[i] = n_randbits(state, k);

            umul_ppmm(p_hi1[i], p_lo1[i], a[i], b[i]);
    
            mulhi_split(&p_hi2[i], a[i], b[i]);
            mullo_split(&p_lo2[i], a[i], b[i]);
        }
        avx2_mullo_split(p_lo3, a, b);
        avx2_mulhi_split(p_hi3, a, b);

        l1 = _nmod_vec_equal(p_lo1, p_lo2, 4); // umul vs mullo
        l2 = _nmod_vec_equal(p_lo1, p_lo3, 4); // umul vs avx2

        h1 = _nmod_vec_equal(p_hi1, p_hi2, 4); // umul vs mulhi
        h2 = _nmod_vec_equal(p_hi1, p_hi3, 4); // umul vs avx2

        printf("nbits=%ld uml=%d ual=%d umh=%d uah=%d\n", k, l1, l2, h1, h2);
    }

    printf("\n");
    for (slong k=59; k <= 64; k++)
    {
        for (slong i = 0; i < 4; i++) 
        {
            a[i] = n_randbits(state, k);
            b[i] = n_randbits(state, k);

            umul_ppmm(p_hi1[i], p_lo1[i], a[i], b[i]);
    
            mulhi_split(&p_hi2[i], a[i], b[i]);
            mullo_split(&p_lo2[i], a[i], b[i]);
        }

        avx2_mullo_split(p_lo3, a, b);
        avx2_mulhi_split(p_hi3, a, b);

        l1 = _nmod_vec_equal(p_lo1, p_lo2, 4);
        l2 = _nmod_vec_equal(p_lo1, p_lo3, 4);

        h1 = _nmod_vec_equal(p_hi1, p_hi2, 4);
        h2 = _nmod_vec_equal(p_hi1, p_hi3, 4);

        printf("nbits=%ld uml=%d ual=%d umh=%d uah=%d\n", k, l1, l2, h1, h2);
    }

    return 0;
}