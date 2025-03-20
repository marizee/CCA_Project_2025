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

#include "../butterfly_fft_64.h"

#define SIZE_MOD 40
#define SIZE_COEFF 39

int main()
{
    slong len = 1 << 25;
    //flint_bitcnt_t bits = 40;
    FLINT_TEST_INIT(state);

    nmod_t mod;
    ulong n, w;
    nn_ptr a, b;
    nn_ptr a_copy1, b_copy1;
    nn_ptr a_copy2, b_copy2;
    nn_ptr a_copy3, b_copy3;
    nn_ptr a_copy4, b_copy4;

    // init modulus structure
    n = n_randbits(state, (uint32_t)SIZE_MOD);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    w = n_randint(state, n);

    // init vector
    a = _nmod_vec_init(len);
    b = _nmod_vec_init(len);
    for (slong i = 0; i < len; i++) 
    {
        //a[i] = n_randbits(state, SIZE_COEFF);
        //if (a[i]==0) a[i]++;
        //b[i] = n_randbits(state, SIZE_COEFF);
        //if (b[i]==0) b[i]++;
        a[i] = n_randint(state, n);
        b[i] = n_randint(state, n);
    }

    // make copies to not overwrite vectors
    a_copy1 = _nmod_vec_init(len);
    b_copy1 = _nmod_vec_init(len);
    _nmod_vec_set(a_copy1, a, len);
    _nmod_vec_set(b_copy1, b, len);

    a_copy2 = _nmod_vec_init(len);
    b_copy2 = _nmod_vec_init(len);
    _nmod_vec_set(a_copy2, a, len);
    _nmod_vec_set(b_copy2, b, len);

    a_copy3 = _nmod_vec_init(len);
    b_copy3 = _nmod_vec_init(len);
    _nmod_vec_set(a_copy3, a, len);
    _nmod_vec_set(b_copy3, b, len);

    a_copy4 = _nmod_vec_init(len);
    b_copy4 = _nmod_vec_init(len);
    _nmod_vec_set(a_copy4, a, len);
    _nmod_vec_set(b_copy4, b, len);


    // print param for debug
    printf("mod.n=%ld, omega=%ld\n", mod.n, w);
    //printf("a=");
    //_nmod_vec_print_pretty(a, len, mod);
    //printf("b=");
    //_nmod_vec_print_pretty(b, len, mod);
    //printf("\n");


    clock_t start, end;
    double tseq, tpreinv, tsplit, tavxsplit;


    start = clock();
    seq_fft(a_copy1, b_copy1, w, len, mod);
    end = clock();
    tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("seq=\t\t%.5es\n", tseq);
    //printf("res_add=");
    //_nmod_vec_print_pretty(a_copy1, len, mod);
    //printf("res_sub=");
    //_nmod_vec_print_pretty(b_copy1, len, mod);
    //printf("\n");


    start = clock();
    preinv_fft(a_copy2, b_copy2, w, len, mod);
    end = clock();
    tpreinv = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("preinv=\t\t%.5es\n", tpreinv);
    //printf("res_add=");
    //_nmod_vec_print_pretty(a_copy2, len, mod);
    //printf("res_sub=");
    //_nmod_vec_print_pretty(b_copy2, len, mod);
    //printf("\n");



    start = clock();
    preinv_split_fft(a_copy3, b_copy3, w, len, mod);
    end = clock();
    tsplit = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("tsplit=\t\t%.5es\n", tsplit);
    //printf("res_add=");
    //_nmod_vec_print_pretty(a_copy3, len, mod);
    //printf("res_sub=");
    //_nmod_vec_print_pretty(b_copy3, len, mod);
    //printf("\n");

    start = clock();
    avx2_preinv_split_fft(a_copy4, b_copy4, w, len, mod);
    end = clock();
    tavxsplit = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("tavxsplit=\t%.5es\n", tavxsplit);
    //printf("res_add=");
    //_nmod_vec_print_pretty(a_copy4, len, mod);
    //printf("res_sub=");
    //_nmod_vec_print_pretty(b_copy4, len, mod);
    //printf("\n");
    

    // checks
    int add1 = _nmod_vec_equal(a_copy1, a_copy2, len);
    int sub1 = _nmod_vec_equal(b_copy1, b_copy2, len);

    int add2 = _nmod_vec_equal(a_copy1, a_copy3, len);
    int sub2 = _nmod_vec_equal(b_copy1, b_copy3, len);

    int add3 = _nmod_vec_equal(a_copy1, a_copy4, len);
    int sub3 = _nmod_vec_equal(b_copy1, b_copy4, len);


    if (!add1 || !sub1 || !add2 || !sub2 || !add3 || !sub3) 
        printf("ff: add1=%d, sub1=%d, add2=%d, sub2=%d, add3=%d, sub3=%d\n", add1, sub1, add2, sub2, add3, sub3);
    else 
        printf("OK!\n");


    _nmod_vec_clear(a); _nmod_vec_clear(b);
    _nmod_vec_clear(a_copy1); _nmod_vec_clear(b_copy1);
    _nmod_vec_clear(a_copy2); _nmod_vec_clear(b_copy2);
    _nmod_vec_clear(a_copy3); _nmod_vec_clear(b_copy3);
    _nmod_vec_clear(a_copy4); _nmod_vec_clear(b_copy4);
    
    FLINT_TEST_CLEAR(state);
    return 0;

}