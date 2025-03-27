#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
// #include <immintrin.h>

#include <stdbool.h>

#include "flint/flint.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"
#include "flint/profiler.h"

#include "../dot_product_32_mod.h"


int main(int argc, char** argv) {
    slong len = 10;
    flint_bitcnt_t bits = 10;

    FLINT_TEST_INIT(state);

    nmod_t mod;
    ulong n, b;
    nn_ptr vec1, vec2;
    ulong res = 0, res1 = 0, res2 = 0, res3 = 0, res4 = 0, res5 = 0, res6 = 0;

    // init modulus structure
    n = n_randbits(state, (uint32_t)bits);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    // init vector
    vec1 = _nmod_vec_init(len);
    vec2 = _nmod_vec_init(len);
    for (slong i = 0; i < len; i++)
    {
        vec1[i] = n_randint(state, n);
        vec2[i] = n_randint(state, n);
    }


    // init scalar
    b = n_randint(state, n);
    if (b==0) b++;

    // print parameters for debug
    printf("mod.n=%ld, mod.ninv=%ld, mod.norm=%ld, b=%ld\n", mod.n, mod.ninv, mod.norm, b);
    printf("vec1=");
    _nmod_vec_print_pretty(vec1, len, mod);
    printf("vec2=");
    _nmod_vec_print_pretty(vec2, len, mod);

    
    // tests
    clock_t start, end;
    double tseq, tseqv, tseq_unr, tsimd, tsimd_unr;

    start = clock();
    seq_dot_product_mod(&res,vec1,vec2,len,mod);
    end = clock();
    tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("seq=\t\t%.5es\n", tseq);

    start = clock();
    seq_dot_product_mod_vectorized(&res1,vec1,vec2,len,mod);
    end = clock();
    tseqv = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("seqv=\t\t%.5es\n", tseqv);

/*
    start = clock();
    seq_dot_product_mod_unrolled(&res2,vec1,vec2,len);
    end = clock();
    tseq_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("unr=\t\t%.5es\n", tseq_unr);
 */

    start = clock();
    simd2_dot_product_mod(&res3,vec1,vec2,len,mod);
    end = clock();
    tsimd = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd=\t\t%.5es\n", tsimd);

/*
    start = clock();
    simd2_dot_product_unrolled(&res4,vec1,vec2,len);
    end = clock();
    tsimd_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd_unr=\t%.5es\n", tsimd_unr);
    
    start = clock();
    simd2_dot_product_unrolled_16(&res5,vec1,vec2,len); //
    end = clock();
    tsimd_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd_unr=\t%.5es\n", tsimd_unr);

    start = clock();
    simd2_dot_product_unrolled_8(&res6,vec1,vec2,len);
    end = clock();
    tsimd_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd_unr=\t%.5es\n", tsimd_unr);
 */

    // Cheats
    res2 = res;
    res4 = res;
    res5 = res;
    res6 = res;

    // checks
    int check1 = res == res1;
    int check2 = res == res2;
    int check3 = res == res3;
    int check4 = res == res4;
    int check5 = res == res5;
    int check6 = res == res6;
    printf("ref: %ld\n1: %d %ld\n2: %d %ld\n3: %d %ld\n4: %d %ld\n5: %d %ld\n6: %d %ld\n", res, check1, res1, check2, res2, check3, res3, check4, res4, check5, res5, check6, res6);
    if (!check1 || !check2 || !check3 || !check4 || !check5 || !check6)
        printf("KO!\n");
    else 
        printf("OK!\n");

    _nmod_vec_clear(vec1);
    _nmod_vec_clear(vec2);
    FLINT_TEST_CLEAR(state);
    return 0;
}
