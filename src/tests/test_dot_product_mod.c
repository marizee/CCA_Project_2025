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
#include "../dot_prod_mod_64.h"


int main(int argc, char** argv) {
    slong len = 10;
    flint_bitcnt_t bits = 50;

    FLINT_TEST_INIT(state);

    nmod_t mod;
    ulong n, b;
    nn_ptr vec1, vec2;
	ulong ref;
    ulong res1 = 0, res2 = 0, res3 = 0, res4 = 0, res5 = 0, res6 = 0, res7 = 0;

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

    ref = _nmod_vec_dot(vec1, vec2, len, mod, _nmod_vec_dot_params(len, mod));

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
    seq_dot_product_mod(&res1,vec1,vec2,len,mod);
    end = clock();
    tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("seq=\t\t%.5es\n", tseq);

    start = clock();
    seq_dot_product_mod_vectorized(&res2,vec1,vec2,len,mod);
    end = clock();
    tseqv = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("seqv=\t\t%.5es\n", tseqv);

    start = clock();
    split_dot_product_mod(&res3,vec1,vec2,len,mod);
    end = clock();
    tsimd_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("split=\t\t%.5es\n", tsimd_unr);
    

    start = clock();
    split_kara_dot_product_mod(&res4,vec1,vec2,len,mod); //
    end = clock();
    tsimd_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("kara=\t\t%.5es\n", tsimd_unr);

    start = clock();
    simd2_dot_product_mod(&res5,vec1,vec2,len,mod);
    end = clock();
    tsimd = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd=\t\t%.5es\n", tsimd);

    start = clock();
    simd2_split_dot_product_mod(&res6,vec1,vec2,len,mod);
    end = clock();
    tseq_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("split_simd=\t%.5es\n", tseq_unr);

    start = clock();
    simd2_kara_dot_product_mod(&res7,vec1,vec2,len,mod);
    end = clock();
    tsimd_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("split_simd=\t%.5es\n", tsimd_unr);

    // checks
    int check1 = ref == res1;
    int check2 = ref == res2;
    int check3 = ref == res3;
    int check4 = ref == res4;
    int check5 = ref == res5;
    int check6 = ref == res6;
    int check7 = ref == res7;
    printf("ref: %ld\n1: %d %ld\n2: %d %ld\n3: %d %ld\n4: %d %ld\n5: %d %ld\n6: %d %ld\n7: %d %ld\n", ref, check1, res1, check2, res2, check3, res3, check4, res4, check5, res5, check6, res6, check7, res7);
    if (!check1 || !check2 || !check3 || !check4 || !check5 || !check6 || !check7)
        printf("KO!\n");
    else 
        printf("OK!\n");

    _nmod_vec_clear(vec1);
    _nmod_vec_clear(vec2);
    FLINT_TEST_CLEAR(state);
    return 0;
}
