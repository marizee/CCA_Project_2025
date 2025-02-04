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
    slong len = 10;
    flint_bitcnt_t bits = 4;
    flint_rand_t state;
    state->__gmp_state = NULL;
    //FLINT_TEST_INIT(state); // ko

    nmod_t mod;
    ulong n, b;
    nn_ptr vec, res, res2, res3;
    

    // init modulus structure
    n = n_randbits(state, (uint32_t)bits);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    printf("n=%ld, mod=%ld\n", n,mod.n);

    // init vector
    vec = _nmod_vec_init(len);
    //_nmod_vec_randtest(vec, state, len, mod);
    for (slong i = 0; i < len; i++)
        vec[i] = n_randint(state, n);
    res = _nmod_vec_init(len);
    res2 = _nmod_vec_init(len);
    res3 = _nmod_vec_init(len);

    // init scalar
    b = n_randint(state, n);
    if (b==0) b++;

    //clock_t start, end;
    //double tseq, tseq_unr;

    //start = clock();
    seq_scalar_vector(res,b,vec,len,mod);
    //end = clock();
    //tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
    //printf("seq=\t%.5es\n", tseq);

    //start = clock();
    seq_scalar_vector_unrolled(res2,b,vec,len,mod);
    //end = clock();
    //tseq_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    //printf("unr=\t%.5es\n", tseq_unr);


    simd_scalar_vector(res3, b, vec, len, mod);

    int result = _nmod_vec_equal(res, res2, len);
    if (!result)
        printf("ff\n");
    else 
        printf("OK! seq == seq_unr\n");

    _nmod_vec_clear(vec);
    _nmod_vec_clear(res);
    _nmod_vec_clear(res2);
    _nmod_vec_clear(res3);
    //FLINT_TEST_CLEAR(state); // ko
    return 0;
}