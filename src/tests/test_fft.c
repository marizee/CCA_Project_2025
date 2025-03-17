#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>

#include <stdbool.h>

#include "flint/flint.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"
#include "flint/profiler.h"

#include "../butterfly_fft.h"

#define SIZE_MOD 40


int main()
{
    slong len = 1 << 20;
    //flint_bitcnt_t bits = 40;
    FLINT_TEST_INIT(state);

    nmod_t mod;
    ulong n, w;
    nn_ptr a, b;
    nn_ptr res_add1, res_sub1;
    nn_ptr res_add2, res_sub2;
    nn_ptr res_add3, res_sub3;

    w = n_randint(state, 30); //1L << (SIZE_MOD-1); //

    // init modulus structure
    n = n_randbits(state, (uint32_t)SIZE_MOD);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    // init vector
    a = _nmod_vec_init(len);
    b = _nmod_vec_init(len);
    for (slong i = 0; i < len; i++) 
    {
        a[i] = n_randint(state, (1L << 32)-1);
        if (a[i]==0) a[i]++;
        b[i] = n_randint(state, (1L << 32)-1);
        if (b[i]==0) b[i]++;
    }
    res_add1 = _nmod_vec_init(len);
    res_sub1 = _nmod_vec_init(len);
    res_add2 = _nmod_vec_init(len);
    res_sub2 = _nmod_vec_init(len);
    res_add3 = _nmod_vec_init(len);
    res_sub3 = _nmod_vec_init(len);



    // print param for debug
    printf("mod.n=%ld, omega=%ld\n", mod.n, w);
    //printf("a=");
    //_nmod_vec_print_pretty(a, len, mod);
    //printf("b=");
    //_nmod_vec_print_pretty(b, len, mod);
    //printf("\n");


    clock_t start, end;
    double tseq, tpreinv, tavx;

    start = clock();
    seq_fft(res_add1, res_sub1, a, b, w, len, mod);
    end = clock();
    tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("seq=\t\t%.5es\n", tseq);
    //printf("res_add=");
    //_nmod_vec_print_pretty(res_add1, len, mod);
    //printf("res_sub=");
    //_nmod_vec_print_pretty(res_sub1, len, mod);
    //printf("\n");

    start = clock();
    preinv_fft(res_add2, res_sub2, a, b, w, len, mod);
    end = clock();
    tpreinv = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("preinv=\t\t%.5es\n", tpreinv);
    //printf("res_add=");
    //_nmod_vec_print_pretty(res_add2, len, mod);
    //printf("res_sub=");
    //_nmod_vec_print_pretty(res_sub2, len, mod);

    start = clock();
    avx2_preinv_fft(res_add3, res_sub3, a, b, w, len, mod);
    end = clock();
    tavx = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("avx=\t\t%.5es\n", tavx);
    //printf("res_add=");
    //_nmod_vec_print_pretty(res_add3, len, mod);
    //printf("res_sub=");
    //_nmod_vec_print_pretty(res_sub3, len, mod);
    //printf("\n");

    /*
    start = clock();
    preinv_fft_unrolled(res_add3, res_sub3, a, b, w, len, mod);
    end = clock();
    tpreunroll = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("preinv_u=\t%.5es\n", tpreunroll);
    //printf("res_add=");
    //_nmod_vec_print_pretty(res_add2, len, mod);
    //printf("res_sub=");
    //_nmod_vec_print_pretty(res_sub2, len, mod);
    */

    // checks
    int add1 = _nmod_vec_equal(res_add1, res_add2, len);
    int sub1 = _nmod_vec_equal(res_sub1, res_sub2, len);

    int add2 = _nmod_vec_equal(res_add1, res_add3, len);
    int sub2 = _nmod_vec_equal(res_sub1, res_sub3, len);

    if (!add1 || !add2 || !sub1 || !sub2) printf("ff: add1=%d, sub1=%d, add2=%d, sub2=%d\n", add1, sub1, add2, sub2);
    else printf("OK!\n");

    _nmod_vec_clear(a);
    _nmod_vec_clear(b);
    _nmod_vec_clear(res_add1);
    _nmod_vec_clear(res_sub1);
    FLINT_TEST_CLEAR(state);
    return 0;
}