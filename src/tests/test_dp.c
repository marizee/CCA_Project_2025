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

#include "../dot_product_32.h"


int main()
{
    slong len = 1 << 14;
    flint_bitcnt_t bits = 25;
    FLINT_TEST_INIT(state);

    //nmod_t mod;
    ulong n;
    nn_ptr vec1, vec2;

    // init modulus structure
    n = n_randbits(state, (uint32_t)bits);
    if (n == UWORD(0)) n++;
    //nmod_init(&mod, n);

    // init vector
    vec1 = _nmod_vec_init(len);
    vec2 = _nmod_vec_init(len);
    for (slong i = 0; i < len; i++)
    {
        vec1[i] = n_randint(state, n);
        vec2[i] = n_randint(state, n);
    }

    // print parameters for debug
    //printf("mod.n=%ld, mod.ninv=%ld, mod.norm=%ld\n", mod.n, mod.ninv, mod.norm);
    //printf("vec1=");
    //_nmod_vec_print_pretty(vec1, len, mod);
    //printf("vec2=");
    //_nmod_vec_print_pretty(vec2, len, mod);

    ulong res1=0, res2=0, res3=0, res4=0, res5=0;
    ulong res8=0, res9=0;
    clock_t start, end;
    double tseq, tseq_v, tseq_unr, tsimd, tsimd_unr;
    double tsplit, tkara;

    start = clock();
    seq_dot_product(&res1,vec1,vec2,len);
    end = clock();
    //printf("\tres\t= %ld\n", res1);
    tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("seq=\t\t%.5es\n", tseq);

    start = clock();
    seq_dot_product_vectorized(&res2,vec1,vec2,len);
    end = clock();
    //printf("res= %ld\n", res2);
    tseq_v = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("seq_v=\t\t%.5es\n", tseq_v);

    start = clock();
    seq_dot_product_unrolled(&res3,vec1,vec2,len);
    end = clock();
    //printf("res= %ld\n", res3);
    tseq_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("seq_unr=\t%.5es\n", tseq_unr);

    start = clock();
    simd2_dot_product(&res4,vec1,vec2,len);
    end = clock();
    //printf("\tres= %ld\n", res4);
    tsimd = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd=\t\t%.5es\n", tsimd);

    start = clock();
    simd2_dot_product_unrolled(&res5,vec1,vec2,len);
    end = clock();
    //printf("res= %ld\n", res5);
    tsimd_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd2_unr=\t%.5es\n", tsimd_unr);

    start = clock();
    split_dot_product(&res8,vec1,vec2,len);
    end = clock();
    //printf("res= %ld\n", res8);
    tsplit = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("split=\t\t%.5es\n", tsplit);

    start = clock();
    split_dot_product(&res9,vec1,vec2,len);
    end = clock();
    //printf("res= %ld\n", res9);
    tkara = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("kara=\t\t%.5es\n", tkara);

#if defined(__AVX512F__)
    ulong res6=0, res7=0;

    double tsimd512, tsimd512_unr;

    start = clock();
    simd512_dot_product(&res6,vec1,vec2,len);
    end = clock();
    //printf("\tres512\t= %ld\n", res6);
    tsimd512 = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd512=\t\t%.5es\n", tsimd512);

    start = clock();
    simd512_dot_product_unrolled(&res7,vec1,vec2,len);
    end = clock();
    //printf("\tres512_unr\t= %ld\n", res7);
    tsimd512_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("simd512_unr=\t%.5es\n", tsimd512_unr);
#endif

    // checks
    // sequentiels
    int check11 = (res1 == res2);
    int check12 = (res2 == res3);

    // simd
    int check2 = (res4 == res5);

    // split
    int check5 = (res8 == res1);
    int check6 = (res9 == res1);
    printf("check5=%d, check6=%d\n", check5, check6);


    // seq vs others
    int check3 = (res3 == res4);


#if defined(__AVX512F__)
    int check22 = (res6 == res7);
    int check4 = (res4 == res6);

    if (!check11 || !check12 || !check2 || !check3 || !check22 || !check4)
    {
	    printf("ff\n");
	//printf("check11=%d; check12=%d; check2=%d; check3=%d; check22=%d; check4=%d\n", check11,check12,check2,check3,check22,check4);
    }else
        printf("OK!\n");
#else
    if (!check11 || !check12 || !check2 || !check3)
        printf("ff\n");
    else
        printf("OK!\n");
#endif
    //printf("check11=%d; check12=%d; check2=%d; check3=%d\n", check11, check12, check2, check3);

    _nmod_vec_clear(vec1);
    _nmod_vec_clear(vec2);
    FLINT_TEST_CLEAR(state);
    return 0;
}
