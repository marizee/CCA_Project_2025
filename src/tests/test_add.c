#include <time.h>
#include "../add_64.h"

// inputs < 2**63

int main()
{
    slong len = 1 << 25;
    //flint_bitcnt_t bits = 63;

    FLINT_TEST_INIT(state);

    nmod_t mod;
    ulong n;
    nn_ptr a, b;
    nn_ptr res1, res2, res3;
    nn_ptr res4, res5;

    // init modulus structure
    //n = n_randbits(state, bits);
    n = 9223372036854775807; // 2**63 - 1
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    // init vectors
    a = _nmod_vec_init(len);
    b = _nmod_vec_init(len);
    for (slong i = 0; i < len; i++)
    {
        a[i] = n_randint(state, n); //9223372036854775807;
        b[i] = n_randint(state, n); //9223372036854775807;
    }
    res1 = _nmod_vec_init(len);
    res2 = _nmod_vec_init(len);
    res3 = _nmod_vec_init(len);
    res4 = _nmod_vec_init(len);
    res5 = _nmod_vec_init(len);

    // print parameters for debug
    //printf("mod.n=%ld\n", mod.n);
    //printf("a=");
    //_nmod_vec_print_pretty(a, len, mod);
    //printf("b=");
    //_nmod_vec_print_pretty(b, len, mod);
    //printf("\n");
    
    // tests
    clock_t start, end;
    double tseq, tseqv, tseq_unr, tsimd2, tsimd2_unr;

    start = clock();
    seq_add(res1,a,b,len);
    end = clock();
    tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res1, len, mod);
    printf("seq=\t\t%.5es\n", tseq);
    //printf("\n");
    

    start = clock();
    seq_add_vectorized(res2,a,b,len);
    end = clock();
    tseqv = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res2, len, mod);
    printf("seqv=\t\t%.5es\n", tseqv);
    //printf("\n");


    start = clock();
    seq_add_unrolled(res3,a,b,len);
    end = clock();
    tseq_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res3, len, mod);
    printf("unr=\t\t%.5es\n", tseq_unr);
    //printf("\n");


    start = clock();
    simd2_add(res4,a,b,len);
    end = clock();
    tsimd2 = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res4, len, mod);
    printf("simd2=\t\t%.5es\n", tsimd2);
    //printf("\n");


    start = clock();
    simd2_add_unrolled(res5,a,b,len);
    end = clock();
    tsimd2_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res5, len, mod);
    printf("simd2_unr=\t%.5es\n", tsimd2_unr);
    //printf("\n");

#if defined(__AVX512F__)
    nn_ptr res6, res7;
    res6 = _nmod_vec_init(len);
    res7 = _nmod_vec_init(len);

    double tsimd512, tsimd512_unr;

    start = clock();
    simd512_add(res6,a,b,len);
    end = clock();
    tsimd512 = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res6, len, mod);
    printf("simd512=\t%.5es\n", tsimd512);
    //printf("\n");

    start = clock();
    simd512_add_unrolled(res7,a,b,len);
    end = clock();
    tsimd512_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res7, len, mod);
    printf("simd512_unr=\t%.5es\n", tsimd512_unr);
    //printf("\n");
#endif
    
    // checks
    int s1 = _nmod_vec_equal(res1, res2, len);
    int s2 = _nmod_vec_equal(res1, res3, len);

    int v2_1 = _nmod_vec_equal(res1, res4, len);
    int v2_2 = _nmod_vec_equal(res1, res5, len);

#if defined(__AVX512F__)
    int v512_1 = _nmod_vec_equal(res1, res6, len);
    int v512_2 = _nmod_vec_equal(res1, res7, len);

    if (!s1 || !s2 || !v2_1 || !v2_2 || !v512_1 || !v512_2)
        printf("ff - s1=%d s2=%d v2_1=%d v2_2=%d v512_1=%d v512_2=%d\n", s1, s2, v2_1, v2_2, v512_1, v512_2);
#else
    if (!s1 || !s2 || !v2_1 || !v2_2)
        printf("ff - s1=%d s2=%d v2_1=%d v2_2=%d\n", s1, s2, v2_1, v2_2);
#endif
    else 
        printf("OK!\n");

    _nmod_vec_clear(a); _nmod_vec_clear(b);
    _nmod_vec_clear(res1);
    _nmod_vec_clear(res2);
    _nmod_vec_clear(res3);
    _nmod_vec_clear(res4);
    _nmod_vec_clear(res5);
#if defined(__AVX512F__)
    _nmod_vec_clear(res6);
    _nmod_vec_clear(res7);
#endif
    FLINT_TEST_CLEAR(state);
    return 0;
}