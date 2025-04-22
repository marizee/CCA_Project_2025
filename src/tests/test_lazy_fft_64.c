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

#include "../lazy_butterfly_fft_64.h"
#include "../mulsplit.h"

#define SIZE_MOD 61

    // requirements:
    //      - n < 2**60
    //      - w < n
    //      - coeffs of a, b < 4*n

// vector equality up to reduction mod
int nmod_vec_red_equal(nn_srcptr vec1, nn_srcptr vec2, ulong len, nmod_t mod)
{
    for (ulong k = 0; k < len; k++)
    {
        ulong v1;
        ulong v2;
        NMOD_RED(v1, vec1[k], mod);
        NMOD_RED(v2, vec2[k], mod);
        if (v1 != v2)
            return 0;
    }

    return 1;
}


int main()
{
    slong len = 1 << 25;
    //flint_bitcnt_t bits = 40;
    FLINT_TEST_INIT(state);

    nmod_t mod;
    ulong n, w, w_pr;
    nn_ptr a, b;
    ulong p_hi, p_lo, tmp;
    nn_ptr a_copy1, b_copy1;
    nn_ptr a_copy2, b_copy2;
    nn_ptr a_copy3, b_copy3;

    // init modulus structure
    n = n_randbits(state, (uint32_t)SIZE_MOD);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    // init w
    w = n_randint(state, n);
    w_pr = n_mulmod_precomp_shoup(w, n);

    // init tmp var
    p_hi=0; p_lo=0; tmp=0;

    // init vector
    a = _nmod_vec_init(len);
    b = _nmod_vec_init(len);
    for (slong i = 0; i < len; i++) 
    {
        a[i] = n_randint(state, 4*n);
        b[i] = n_randint(state, 4*n);
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

    clock_t start, end;
    double tseq, tavx2;

    // print param for debug
    printf("mod.n=%ld, omega=%ld\n", mod.n, w);
    //printf("a=");
    //_nmod_vec_print_pretty(a, len, mod);
    //printf("b=");
    //_nmod_vec_print_pretty(b, len, mod);
    //printf("\n");

    start = clock();
    preinv_fft_lazy44(a_copy1,b_copy1,w,w_pr,len,n,2*n,p_hi,p_lo,tmp);
    end = clock();
    tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("seq=\t\t%.5es\n", tseq);
    //printf("add=");
    //_nmod_vec_print_pretty(a_copy1, len, mod);
    //printf("sub=");
    //_nmod_vec_print_pretty(b_copy1, len, mod);
    //printf("\n");

    start = clock();
    avx2_preinv_split_fft_lazy44(a_copy2,b_copy2,w,w_pr,len,n,2*n,p_hi,p_lo,tmp);
    end = clock();
    tavx2 = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("avx2=\t\t%.5es\n", tavx2);
    //printf("add=");
    //_nmod_vec_print_pretty(a_copy2, len, mod);
    //printf("sub=");
    //_nmod_vec_print_pretty(b_copy2, len, mod);
    //printf("\n");

    int add1 = nmod_vec_red_equal(a_copy1, a_copy2, len, mod);
    int sub1 = nmod_vec_red_equal(b_copy1, b_copy2, len, mod);

#if defined(__AVX512F__)

    double tavx512;

    start = clock();
    avx512_preinv_split_fft_lazy44(a_copy3,b_copy3,w,w_pr,len,n,2*n,p_hi,p_lo,tmp);
    end = clock();
    tavx512 = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("avx512=\t\t%.5es\n", tavx512);
    //printf("add=");
    //_nmod_vec_print_pretty(a_copy3, len, mod);
    //printf("sub=");
    //_nmod_vec_print_pretty(b_copy3, len, mod);
    //printf("\n");

    int add2 = nmod_vec_red_equal(a_copy1, a_copy3, len, mod);
    int sub2 = nmod_vec_red_equal(b_copy1, b_copy3, len, mod);

    printf("add1=%d, sub1=%d, add2=%d, sub2=%d\n", add1, sub1, add2, sub2);
#else
    printf("add1=%d, sub1=%d\n", add1, sub1);

#endif
    _nmod_vec_clear(a); _nmod_vec_clear(b);
    _nmod_vec_clear(a_copy1); _nmod_vec_clear(b_copy1);
    _nmod_vec_clear(a_copy2); _nmod_vec_clear(b_copy2);
    _nmod_vec_clear(a_copy3); _nmod_vec_clear(b_copy3);
    
    FLINT_TEST_CLEAR(state);

    return 0;
}
