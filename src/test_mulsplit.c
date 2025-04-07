#include "mulsplit.h"

// For now, only checks the correctness of the functions.

int main()
{
    FLINT_TEST_INIT(state);
    slong len = 8;

    nn_ptr a, b;
    nn_ptr p_lo1, p_hi1;
    nn_ptr p_lo2, p_hi2;
    nn_ptr p_lo3, p_hi3;
    nn_ptr p_lo4;
#if defined(__AVX512F__)
    nn_ptr p_lo5, p_hi5;
#endif

    // init vector
    a = _nmod_vec_init(len); b = _nmod_vec_init(len);
    p_lo1 = _nmod_vec_init(len); p_hi1 = _nmod_vec_init(len);   // umul
    p_lo2 = _nmod_vec_init(len); p_hi2 = _nmod_vec_init(len);   // seq
    p_lo3 = _nmod_vec_init(len); p_hi3 = _nmod_vec_init(len);   // avx2 mine
    p_lo4 = _nmod_vec_init(len);                                // avx2 stack
#if defined(__AVX512F__)
    p_lo5 = _nmod_vec_init(len); p_hi5 = _nmod_vec_init(len);   // avx512 mullo; mine
#endif

    for (slong k=25; k < 64; k++)
    {
        // init vectors
        for (slong i = 0; i < len; i++) 
        {
            a[i] = n_randbits(state, k);
            b[i] = n_randbits(state, k);
        }

        // seq versions
        for (slong i = 0; i < len; i++)
        {
            umul_ppmm(p_hi1[i], p_lo1[i], a[i], b[i]);

            p_lo2[i] = mullo_split(a[i], b[i]);
            p_hi2[i] = mulhi_split(a[i], b[i]);
        }

        // avx2 versions
        for (slong i = 0; i+3 < len; i+=4)
        {
            __m256i va = _mm256_loadu_si256((const __m256i *)(a+i));
            __m256i vb = _mm256_loadu_si256((const __m256i *)(b+i));
            __m256i low = _mm256_setzero_si256();
            __m256i high = _mm256_setzero_si256();

            low = avx2_mullo_split(va, vb);
            high = avx2_mulhi_split(va, vb);
            _mm256_storeu_si256((__m256i *)(p_lo3+i), low);
            _mm256_storeu_si256((__m256i *)(p_hi3+i), high);

            low = avx2_mullo_epi64(va, vb);
            _mm256_storeu_si256((__m256i *)(p_lo4+i), low);
        }

        // avx512 versions
#if defined(__AVX512F__)
        for (slong i = 0; i+7 < len; i+=8)
        {
            __m512i va = _mm512_loadu_si512((const __m512i *)(a+i));
            __m512i vb = _mm512_loadu_si512((const __m512i *)(b+i));
            __m512i low = _mm512_setzero_si512();
            __m512i high = _mm512_setzero_si512();

            low = _mm512_mullo_epi64(va, vb);
            _mm512_storeu_si512((__m512i *)(p_lo5+i), low);

            high = avx512_mulhi_split_v2(va, vb);
            _mm512_storeu_si512((__m512i *)(p_hi5+i), high);
        }
#endif

        // checks
        int l1, l2, l3;
        int h1, h2;

        l1 = _nmod_vec_equal(p_lo1, p_lo2, 4); // low: umul vs mullo
        l2 = _nmod_vec_equal(p_lo1, p_lo3, 4); // low: umul vs avx2 mine
        l3 = _nmod_vec_equal(p_lo1, p_lo4, 4); // low: umul vs avx2 stack

        h1 = _nmod_vec_equal(p_hi1, p_hi2, 4); // high: umul vs mulhi
        h2 = _nmod_vec_equal(p_hi1, p_hi3, 4); // high: umul vs avx2

#if defined(__AVX512F__)
        int l4;
        int h3;

        l4 = _nmod_vec_equal(p_lo1, p_lo5, 4); // low: umul vs avx512 mullo
        h3 = _nmod_vec_equal(p_hi1, p_hi5, 4); // high: umul vs avx512 mine

        printf("nbits=%ld -- l1=%d l2=%d l3=%d l4=%d -- h1=%d h2=%d h3=%d\n", k, l1, l2, l3, l4, h1, h2, h3);
#else
        printf("nbits=%ld -- l1=%d l2=%d l3=%d -- h1=%d h2=%d\n", k, l1, l2, l3, h1, h2);
#endif
    }


    _nmod_vec_clear(a); _nmod_vec_clear(b); 
    _nmod_vec_clear(p_lo1); _nmod_vec_clear(p_hi1);
    _nmod_vec_clear(p_lo2); _nmod_vec_clear(p_hi2);
    _nmod_vec_clear(p_lo3); _nmod_vec_clear(p_hi3);
    _nmod_vec_clear(p_lo4);
#if defined(__AVX512F__)
    _nmod_vec_clear(p_lo5); _nmod_vec_clear(p_hi5);
#endif
    FLINT_TEST_CLEAR(state);

    return 0;
}
