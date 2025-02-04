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

#include "scalar_vector_32.h"


int main()
{
    slong len = 2048;
    flint_bitcnt_t bits = 31;
    flint_rand_t state;
    state->__gmp_state = NULL;
    //FLINT_TEST_INIT(state); // ko

    nmod_t mod;
    ulong n, b;
    nn_ptr vec, res, res2, res3, res4;

    // init modulus structure
    n = n_randbits(state, (uint32_t)bits);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    // init vector
    vec = _nmod_vec_init(len);
    for (slong i = 0; i < len; i++)
        vec[i] = n_randint(state, n);
    res = _nmod_vec_init(len);
    res2 = _nmod_vec_init(len);
    res3 = _nmod_vec_init(len);
    res4 = _nmod_vec_init(len);


    // init scalar
    b = n_randint(state, n);
    if (b==0) b++;

    // print parameters for debug
    //printf("mod=%ld, b=%ld\n", mod.n, b);
    //printf("vec=");
    //_nmod_vec_print_pretty(vec, len, mod);

    
    // tests
    clock_t start, end;
    double tseq, tseq_unr, tsimd;

    start = clock();
    seq_scalar_vector(res,b,vec,len,mod);
    end = clock();
    tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("seq=\t%.5es\n", tseq);

    start = clock();
    seq_scalar_vector_unrolled(res2,b,vec,len,mod);
    end = clock();
    tseq_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("unr=\t%.5es\n", tseq_unr);

    start = clock();
    simd_scalar_vector(res3, b, vec, len, mod);
    //_nmod_vec_print_pretty(res3, len, mod);
    end = clock();
    tsimd = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd=\t%.5es\n", tsimd);

    //simd_scalar_vector_unrolled(res4, b, vec, len, mod);
    //_nmod_vec_print_pretty(res4, len, mod);

    
    // checks
    int check1 = _nmod_vec_equal(res, res2, len);
    int check2 = 1;//_nmod_vec_equal(res2, res3, len);
    if (!check1 || !check2)
        printf("ff\n");
    else 
        printf("OK! seq == seq_unr\n");

    _nmod_vec_clear(vec);
    _nmod_vec_clear(res);
    _nmod_vec_clear(res2);
    _nmod_vec_clear(res3);
    _nmod_vec_clear(res4);
    //FLINT_TEST_CLEAR(state); // ko
    return 0;
}