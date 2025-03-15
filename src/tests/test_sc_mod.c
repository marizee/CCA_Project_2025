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

#include "../scalar_vector_mod_32.h"


int main()
{
    slong len = 1 << 10;
    flint_bitcnt_t bits = 45;

    FLINT_TEST_INIT(state);

    nmod_t mod;
    ulong n, b;
    nn_ptr vec, res, res1, res2, res3, res4;

    // init modulus structure
    n = n_randbits(state, (uint32_t)bits);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    // init vector
    vec = _nmod_vec_init(len);
    for (slong i = 0; i < len; i++)
        vec[i] = n_randint(state, 32);
    res = _nmod_vec_init(len);
    res1 = _nmod_vec_init(len);
    res2 = _nmod_vec_init(len);
    res3 = _nmod_vec_init(len);
    res4 = _nmod_vec_init(len);


    // init scalar
    b = n_randint(state, 32);
    if (b==0) b++;

    // print parameters for debug
    printf("b=%ld\n", b);
    //printf("mod.n=%ld, mod.ninv=%ld, mod.norm=%ld, b=%ld\n", mod.n, mod.ninv, mod.norm, b);
    //printf("vec=");
    //_nmod_vec_print_pretty(vec, len, mod);

    
    // tests
    clock_t start, end;
    double tseq, tseqv, tseq_unr, tsimd, tsimd_unr;

    start = clock();
    seq_mod_scalar_vector(res,b,vec,len,mod);
    end = clock();
    tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res, len, mod);
    printf("seq=\t\t%.5es\n", tseq);

    start = clock();
    seq_mod_scalar_vector_vectorized(res1,b,vec,len,mod);
    end = clock();
    tseqv = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res1, len, mod);
    printf("seqv=\t\t%.5es\n", tseqv);

    start = clock();
    seq_mod_scalar_vector_unrolled(res2,b,vec,len,mod);
    end = clock();
    tseq_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    //_nmod_vec_print_pretty(res2, len, mod);
    printf("unr=\t\t%.5es\n", tseq_unr);

    start = clock();
    simd2_mod_scalar_vector(res3, b, vec, len, mod);
    end = clock();
    //_nmod_vec_print_pretty(res3, len, mod);
    tsimd = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd=\t\t%.5es\n", tsimd);
/*
    start = clock();
    simd2_mod_scalar_vector_unrolled(res4, b, vec, len);
    //_nmod_vec_print_pretty(res4, len, mod);
    end = clock();
    tsimd_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd_unr=\t%.5es\n", tsimd_unr);
    

*/
    // checks
    int check1 = _nmod_vec_equal(res, res2, len);
    int check11 = _nmod_vec_equal(res1, res2, len);
    //int check2 = _nmod_vec_equal(res3, res4, len);
    int check3 = _nmod_vec_equal(res, res3, len);
    
    if (!check1 || !check11 || !check3)
        printf("ff\n");
    else 
        printf("OK!\n");

    _nmod_vec_clear(vec);
    _nmod_vec_clear(res);
    _nmod_vec_clear(res1);
    _nmod_vec_clear(res2);
    _nmod_vec_clear(res3);
    _nmod_vec_clear(res4);
    FLINT_TEST_CLEAR(state);
    return 0;
}