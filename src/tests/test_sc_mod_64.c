#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>

#include <stdbool.h>

#include "../scalar_vector_mod_64.h"
#include "../mulsplit.h"


int main()
{
    slong len = 1 << 20;
    flint_bitcnt_t bits = 60;

    FLINT_TEST_INIT(state);

    nmod_t mod;
    ulong n, w;
    nn_ptr b, res1, res2, res3, res4;
    nn_ptr res5, res6;

    // init modulus structure
    n = n_randbits(state, (uint32_t)bits);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    // init vector
    b = _nmod_vec_init(len);
    for (slong i = 0; i < len; i++)
        b[i] = n_randint(state, n);
    res1 = _nmod_vec_init(len);
    res2 = _nmod_vec_init(len);
    res3 = _nmod_vec_init(len);
    res4 = _nmod_vec_init(len);
    res5 = _nmod_vec_init(len);
    res6 = _nmod_vec_init(len);

    // init scalar
    w = n_randint(state, n);
    if (w==0) w++;

    // print parameters for debug
    printf("mod=%ld; omega=%ld\n", mod.n, w);
    //printf("vec=");
    //_nmod_vec_print_pretty(b, len, mod);
    printf("\n");

    
    // tests
    clock_t start, end;
    double tflint, tseq, tseqv, tseq_unr;
    double tavx2, tavx2_unr;

    start = clock();
    flint_shoup_scalar_vector_mod_64(res1,w,b,len,mod);
    end = clock();
    tflint = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res1, len, mod);
    printf("tflint=\t\t%.5es\n", tflint);

    start = clock();
    seq_shoup_scalar_vector_mod_64(res2,w,b,len,mod);
    end = clock();
    tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res2, len, mod);
    printf("seq=\t\t%.5es\n", tseq);

    start = clock();
    seq_shoup_scalar_vector_mod_64_vectorized(res3,w,b,len,mod);
    end = clock();
    tseqv = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res3, len, mod);
    printf("seqv=\t\t%.5es\n", tseqv);

    start = clock();
    seq_shoup_scalar_vector_mod_64_unrolled(res4,w,b,len,mod);
    end = clock();
    tseq_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res4, len, mod);
    printf("unr=\t\t%.5es\n", tseq_unr);

    start = clock();
    avx2_shoup_scalar_vector_mod_64(res5,w,b,len,mod);
    end = clock();
    //_nmod_vec_print_pretty(res5, len, mod);
    tavx2 = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("avx2=\t\t%.5es\n", tavx2);

    start = clock();
    avx2_shoup_scalar_vector_mod_64_unrolled(res6,w,b,len,mod);
    //_nmod_vec_print_pretty(res6, len, mod);
    end = clock();
    tavx2_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("avx2_unr=\t%.5es\n", tavx2_unr);
    

#if defined(__AVX512F__)
    nn_ptr res7, res8;
    res7 = _nmod_vec_init(len);
    res8 = _nmod_vec_init(len);

    double tavx512, tavx512_unr;

    start = clock();
    avx512_shoup_scalar_vector_mod_64(res7,w,b,len,mod);
    end = clock();
    //_nmod_vec_print_pretty(res7, len, mod);
    tavx512 = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("avx512=\t\t%.5es\n", tavx512);

    start = clock();
    avx512_shoup_scalar_vector_mod_64_unrolled(res8,w,b,len,mod);
    //_nmod_vec_print_pretty(res8, len, mod);
    end = clock();
    tavx512_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("avx512_unr=\t%.5es\n", tavx512_unr);

#endif

    // checks
    int s1 = _nmod_vec_equal(res1, res2, len);
    int s2 = _nmod_vec_equal(res1, res3, len);
    int s3 = _nmod_vec_equal(res1, res4, len);
    
    int v2_1 = _nmod_vec_equal(res1, res5, len);
    int v2_2 = _nmod_vec_equal(res1, res6, len);

#if defined(__AVX512F__)
    int v512_1 = _nmod_vec_equal(res1, res7, len);
    int v512_2 = _nmod_vec_equal(res1, res8, len);

    if (!s1 || !s2 || !s3 || !v2_1 || !v2_2 || !v512_1 || !v512_2)
        printf("ff - s1=%d s2=%d s3=%d v2_1=%d v2_2=%d v512_1=%d v512_2=%d\n", s1, s2, s3, v2_1, v2_2, v512_1, v512_2);
#else
    if (!s1 || !s2 || !s3 || !v2_1 || !v2_2)
        printf("ff - s1=%d s2=%d s3=%d v2_1=%d v2_2=%d\n", s1, s2, s3, v2_1, v2_2);
#endif
    else 
        printf("OK!\n");

    _nmod_vec_clear(b);
    _nmod_vec_clear(res1);
    _nmod_vec_clear(res2);
    _nmod_vec_clear(res3);
    _nmod_vec_clear(res4);
    _nmod_vec_clear(res5);
    _nmod_vec_clear(res6);
#if defined(__AVX512F__)
    _nmod_vec_clear(res7);
    _nmod_vec_clear(res8);
#endif
    FLINT_TEST_CLEAR(state);
    return 0;
}
