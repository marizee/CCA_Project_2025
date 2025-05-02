#include <time.h>
#include "../dot_prod_mod_64.h"


int main()
{
    slong len = 1 << 25;
    flint_bitcnt_t bits = 60;

    FLINT_TEST_INIT(state);

    nmod_t mod;
    ulong n;
    nn_ptr a, b;
    ulong res1, res2, res3, res4;
    ulong res5;

    // init modulus structure
    n = n_randbits(state, bits);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    // init vectors
    a = _nmod_vec_init(len);
    b = _nmod_vec_init(len);
    for (slong i = 0; i < len; i++)
    {
        a[i] = n_randint(state, n);
        b[i] = n_randint(state, n);
    }
    res1 = 0;
    res2 = 0;
    res3 = 0;
    res4 = 0;
    res5 = 0;

    // print parameters for debug
    printf("mod.n=%ld\n", mod.n);
    //printf("a=");
    //_nmod_vec_print_pretty(a, len, mod);
    //printf("b=");
    //_nmod_vec_print_pretty(b, len, mod);
    //printf("\n");
    
    // tests
    clock_t start, end;
    double tflint, tseq, tseq_unr, tseqv, tsimd2;

    start = clock();
    flint_dot_prod_mod_64(&res1,a,b,len,mod);
    end = clock();
    tflint = ((double) (end - start)) / CLOCKS_PER_SEC;
    //printf("res=%ld\n",res1);
    printf("flint=\t\t%.5es\n", tflint);
    //printf("\n");
    

    start = clock();
    seq_dot_prod_mod_64(&res2,a,b,len,mod);
    end = clock();
    tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
    //printf("res=%ld\n",res2);
    printf("seq=\t\t%.5es\n", tseq);
    //printf("\n");

    start = clock();
    seq_dot_prod_mod_64_vectorized(&res3,a,b,len,mod);
    end = clock();
    tseqv = ((double) (end - start)) / CLOCKS_PER_SEC;
    //printf("res=%ld\n",res3);
    printf("seqv=\t\t%.5es\n", tseqv);
    //printf("\n");


    start = clock();
    seq_dot_prod_mod_64_unrolled(&res4,a,b,len,mod);
    end = clock();
    tseq_unr = ((double) (end - start)) / CLOCKS_PER_SEC;
    //printf("res=%ld\n", res4);
    printf("unr=\t\t%.5es\n", tseq_unr);
    //printf("\n");

    start = clock();
    simd2_dot_prod_mod_64(&res5,a,b,len,mod);
    end = clock();
    tsimd2 = ((double) (end - start)) / CLOCKS_PER_SEC;
    ////printf("res=%ld\n", res5);
    //printf("simd2=\t\t%.5es\n", tsimd2);
    ////printf("\n");


    // checks
    int s1 = (res1 == res2);
    int s2 = (res1 == res3);
    int s3 = (res1 == res4);

    //int v2_1 = (res1 == res5);

    if (!s1 || !s2 || !s3)
        printf("ff - s1=%d s2=%d s3=%d\n", s1, s2, s3);
    else
        printf("OK!\n");

    _nmod_vec_clear(a); _nmod_vec_clear(b);
    FLINT_TEST_CLEAR(state);
    return 0;
}