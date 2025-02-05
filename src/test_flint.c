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
    slong len = 1 << 27;
    flint_bitcnt_t bits = 31;
    FLINT_TEST_INIT(state);

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
    //printf("mod.n=%ld, mod.ninv=%ld, mod.norm=%ld, b=%ld\n", mod.n, mod.ninv, mod.norm, b);
    //printf("50 mod n=%ld\n", 50*mod.ninv);
    //printf("vec=");
    //_nmod_vec_print_pretty(vec, len, mod);

    
    // tests
    clock_t start, end;
    double tseq, tseq_unr, tsimd, tsimd_unr;

    start = clock();
    seq_scalar_vector(res,b,vec,len,mod);
    end = clock();
    tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("seq=\t\t%.5es\n", tseq);

    start = clock();
    seq_scalar_vector_unrolled(res2,b,vec,len,mod);
    end = clock();
    tseq_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("unr=\t\t%.5es\n", tseq_unr);

    start = clock();
    simd_scalar_vector(res3, b, vec, len, mod);
    //_nmod_vec_print_pretty(res3, len, mod);
    end = clock();
    tsimd = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd=\t\t%.5es\n", tsimd);

    start = clock();
    simd_scalar_vector_unrolled(res4, b, vec, len, mod);
    //_nmod_vec_print_pretty(res4, len, mod);
    end = clock();
    tsimd_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd_unr=\t%.5es\n", tsimd_unr);
    
    // checks
    int check1 = _nmod_vec_equal(res, res2, len);
    int check2 = _nmod_vec_equal(res3, res4, len);
    if (!check1 || !check2)
        printf("ff\n");
    else 
        printf("OK!\n");

    _nmod_vec_clear(vec);
    _nmod_vec_clear(res);
    _nmod_vec_clear(res2);
    _nmod_vec_clear(res3);
    _nmod_vec_clear(res4);
    FLINT_TEST_CLEAR(state);
    return 0;
}