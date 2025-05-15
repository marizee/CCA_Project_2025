#include "flint/profiler.h"
#include "flint/ulong_extras.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/flint.h"

#include "scalar_vector_32.h"
#include "scalar_vector_mod_32.h"
#include "scalar_vector_mod_64.h"
#include "dot_product_32.h"
#include "dot_product_32_mod.h"
#include "dot_prod_mod_64.h"
#include "butterfly_fft_64.h"
#include "lazy_butterfly_fft_64.h"
#include "add_64.h"
#include "add_mod_64.h"

#include <string.h>


typedef struct
{
   flint_bitcnt_t bits;
   slong length;
} info_t;

void add_64(info_t* info, void (*func)(nn_ptr, nn_ptr, nn_ptr, slong))
{
    // retrieve function parameters
    //info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;
    
    FLINT_TEST_INIT(state);

    //nmod_t mod;
    ulong n;
    nn_ptr a, b;
    nn_ptr res;

    // init modulus structure
    n = n_randbits(state, bits);
    if (n == UWORD(0)) n++;
    //nmod_init(&mod, n);

    // init vectors
    a = _nmod_vec_init(length);
    b = _nmod_vec_init(length);
    for (slong i = 0; i < length; i++)
    {
        a[i] = n_randint(state, n);
        b[i] = n_randint(state, n);
    }
    res = _nmod_vec_init(length);
    
    double FLINT_SET_BUT_UNUSED(tcpu), twall;

    TIMEIT_START
    func(res,a,b,length);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);
    
    _nmod_vec_clear(a); _nmod_vec_clear(b);
    _nmod_vec_clear(res);
    FLINT_TEST_CLEAR(state);
}

void add_mod_64(info_t* info, void (*func)(nn_ptr, nn_ptr, nn_ptr, slong, nmod_t))
{
    // retrieve function parameters
    //info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;
    
    FLINT_TEST_INIT(state);

    nmod_t mod;
    ulong n;
    nn_ptr a, b;
    nn_ptr res;

    // init modulus structure
    n = n_randbits(state, bits);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    // init vectors
    a = _nmod_vec_init(length);
    b = _nmod_vec_init(length);
    for (slong i = 0; i < length; i++)
    {
        a[i] = n_randint(state, n);
        b[i] = n_randint(state, n);
    }
    res = _nmod_vec_init(length);
    
    double FLINT_SET_BUT_UNUSED(tcpu), twall;

    TIMEIT_START
    func(res,a,b,length,mod);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);
    
    _nmod_vec_clear(a); _nmod_vec_clear(b);
    _nmod_vec_clear(res);
    FLINT_TEST_CLEAR(state);
}

void scalar_vector(info_t* info, void (*func)(nn_ptr, ulong, nn_ptr, slong))
{
    // retrieve function parameters
    //info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;
    
    FLINT_TEST_INIT(state);
    
    //nmod_t mod;
    ulong n, b;
    nn_ptr vec, res;
    //ulong i;
    slong j;

    vec = _nmod_vec_init(length);
    res = _nmod_vec_init(length);

    
    // init modulus
    n = n_randbits(state, (uint32_t)bits);
    if (n == UWORD(0)) n++;
    //nmod_init(&mod, n);
    
    // init scalar and vector
    b = n_randint(state, n);
    for (j = 0; j < length; j++)
        vec[j] = n_randint(state, n);
    
    double FLINT_SET_BUT_UNUSED(tcpu), twall;

    TIMEIT_START
    func(res,b,vec,length);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);
    
    _nmod_vec_clear(res);
    _nmod_vec_clear(vec);
    FLINT_TEST_CLEAR(state);
}

void scalar_vector_mod(info_t* info, void (*func)(nn_ptr, ulong, nn_ptr, slong, nmod_t))
{
    // retrieve function parameters
    //info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;
    
    FLINT_TEST_INIT(state);
    
    nmod_t mod;
    ulong n, b;
    nn_ptr vec, res;
    //ulong i;
    slong j;

    vec = _nmod_vec_init(length);
    res = _nmod_vec_init(length);

    
    // init modulus
    n = n_randbits(state, (uint32_t)bits);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);
    
    // init scalar and vector
    b = n_randint(state, 32);
    if (b==0) b++;
    for (j = 0; j < length; j++)
        vec[j] = n_randint(state, 32);
    
    double FLINT_SET_BUT_UNUSED(tcpu), twall;

    TIMEIT_START
    func(res,b,vec,length,mod);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);
    
    _nmod_vec_clear(res);
    _nmod_vec_clear(vec);
    FLINT_TEST_CLEAR(state);
}

void scalar_vector_mod_64(info_t* info, void (*func)(nn_ptr, ulong, nn_ptr, slong, nmod_t))
{
    // retrieve function parameters
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;
    
    FLINT_TEST_INIT(state);

    nmod_t mod;
    ulong n, w;
    nn_ptr b, res;

    // init modulus structure
    n = n_randbits(state, (uint32_t)bits);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    // init vector
    b = _nmod_vec_init(length);
    for (slong i = 0; i < length; i++)
        b[i] = n_randint(state, n);
    res = _nmod_vec_init(length);

    // init scalar
    w = n_randint(state, n);
    if (w==0) w++;

    double FLINT_SET_BUT_UNUSED(tcpu), twall;

    TIMEIT_START
    func(res,w,b,length,mod);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);

    _nmod_vec_clear(b);
    _nmod_vec_clear(res);
    FLINT_TEST_CLEAR(state);

}

void dot_prod(info_t* info, void (*func)(ulong*, nn_ptr, nn_ptr, slong))
{
    // retrieve function parameters
    //info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;
    
    FLINT_TEST_INIT(state);
    
    //nmod_t mod;
    ulong n;
    nn_ptr vec1, vec2;
    //ulong i;
    slong j;

    vec1 = _nmod_vec_init(length);
    vec2 = _nmod_vec_init(length);
    ulong res=0;
    
    // init modulus
    n = n_randbits(state, (uint32_t)bits);
    if (n == UWORD(0)) n++;
    //nmod_init(&mod, n);
    
    // init scalar and vector
    //b = n_randint(state, n);
    for (j = 0; j < length; j++)
    {
        vec1[j] = n_randint(state, n);
        vec2[j] = n_randint(state, n);
    }
    
    double FLINT_SET_BUT_UNUSED(tcpu), twall;
    TIMEIT_START
    func(&res,vec1,vec2,length);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);
    
    _nmod_vec_clear(vec1);
    _nmod_vec_clear(vec2);
    FLINT_TEST_CLEAR(state);
}

void dot_prod_mod(info_t* info, void (*func)(ulong*, nn_ptr, nn_ptr, slong, nmod_t))
{
    // retrieve function parameters
    //info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;
    
    FLINT_TEST_INIT(state);
    
    nmod_t mod;
    ulong n;
    nn_ptr vec1, vec2;
    //ulong i;
    slong j;

    vec1 = _nmod_vec_init(length);
    vec2 = _nmod_vec_init(length);
    ulong res=0;
    
    // init modulus
    n = n_randbits(state, (uint32_t)bits);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);
    
    // init scalar and vector
    //b = n_randint(state, n);
    for (j = 0; j < length; j++)
    {
        vec1[j] = n_randint(state, n);
        vec2[j] = n_randint(state, n);
    }
    
    double FLINT_SET_BUT_UNUSED(tcpu), twall;
    TIMEIT_START
    func(&res,vec1,vec2,length, mod);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);
    
    _nmod_vec_clear(vec1);
    _nmod_vec_clear(vec2);
    FLINT_TEST_CLEAR(state);
}

void butterfly_fft_64(info_t* info, void (*func)(nn_ptr, nn_ptr, ulong, slong, nmod_t))
{
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;

    FLINT_TEST_INIT(state);
    
    nmod_t mod;
    ulong n, w;
    nn_ptr a, b;
    nn_ptr a_copy, b_copy;

    // init modulus structure
    n = n_randbits(state, (uint32_t)bits);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    w = n_randint(state, n);

    // init vector
    a = _nmod_vec_init(length);
    b = _nmod_vec_init(length);
    for (slong i = 0; i < length; i++) 
    {
        a[i] = n_randint(state, n);
        b[i] = n_randint(state, n);
    }

    // make copies to not overwrite vectors
    a_copy = _nmod_vec_init(length);
    b_copy = _nmod_vec_init(length);
    _nmod_vec_set(a_copy, a, length);
    _nmod_vec_set(b_copy, b, length);

    double FLINT_SET_BUT_UNUSED(tcpu), twall;

    TIMEIT_START
    func(a_copy,b_copy,w,length,mod);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);
    
    _nmod_vec_clear(a);
    _nmod_vec_clear(b);
    _nmod_vec_clear(a_copy);
    _nmod_vec_clear(b_copy);
    FLINT_TEST_CLEAR(state);
}

void lazy_butterfly_64(info_t* info, void (*func)(nn_ptr, nn_ptr, ulong, ulong, slong, ulong, ulong, ulong, ulong, ulong))
{
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;
    FLINT_TEST_INIT(state);

    nmod_t mod;
    ulong n, w, w_pr;
    nn_ptr a, b;
    ulong p_hi, p_lo, tmp;
    nn_ptr a_copy, b_copy;

    // init modulus structure
    n = n_randbits(state, (uint32_t)bits);
    if (n == UWORD(0)) n++;
    nmod_init(&mod, n);

    // init w
    w = n_randint(state, n);
    w_pr = n_mulmod_precomp_shoup(w, n);

    // init tmp var
    p_hi=0; p_lo=0; tmp=0;

    // init vector
    a = _nmod_vec_init(length);
    b = _nmod_vec_init(length);
    for (slong i = 0; i < length; i++) 
    {
        a[i] = n_randint(state, 4*n);
        b[i] = n_randint(state, 4*n);
    }

    // make copy to not overwrite vectors
    a_copy = _nmod_vec_init(length);
    b_copy = _nmod_vec_init(length);
    _nmod_vec_set(a_copy, a, length);
    _nmod_vec_set(b_copy, b, length);

    double FLINT_SET_BUT_UNUSED(tcpu), twall;

    TIMEIT_START
    func(a_copy,b_copy,w,w_pr,length,n,2*n,p_hi,p_lo,tmp);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);
    
    _nmod_vec_clear(a);
    _nmod_vec_clear(b);
    _nmod_vec_clear(a_copy);
    _nmod_vec_clear(b_copy);
    FLINT_TEST_CLEAR(state);
}


int main(int argc, char** argv)
{
    info_t info;
    slong len;

    typedef void (*func) ();
    typedef void (*timefun) (info_t*, func);
    const timefun funs[] = {
        add_64,
        add_mod_64,
        scalar_vector,
        scalar_vector_mod,
        scalar_vector_mod_64,
        dot_prod,
        dot_prod_mod,
        dot_prod_mod,
        butterfly_fft_64,
        lazy_butterfly_64,
    };

    // all versions of the function
    const func versions[][15] = {
    {   // add 64
        seq_add,
        seq_add_vectorized,
        seq_add_unrolled,
        simd2_add,
        simd2_add_unrolled,
#if defined(__AVX512F__)
        simd512_add,
        simd512_add_unrolled,
#endif
    },
    {   // add mod 64
        seq_add_mod,
        seq_add_mod_vectorized,
        seq_add_mod_unrolled,    
        simd2_add_mod,
        simd2_add_mod_unrolled,
#if defined(__AVX512F__)
        simd512_add_mod,
        simd512_add_mod_unrolled,
#endif
    },
    {   // scalar vector 32
        seq_scalar_vector,
        seq_scalar_vector_vectorized,
        seq_scalar_vector_unrolled,      
        simd2_scalar_vector,
        simd2_scalar_vector_unrolled,
#if defined(__AVX512F__)
        simd512_scalar_vector,
        simd512_scalar_vector_unrolled,
#endif
    },
    {   // scalar vector mod 32
        seq_mod_scalar_vector,
        seq_mod_scalar_vector_vectorized,
        seq_mod_scalar_vector_unrolled,
        simd2_mod_scalar_vector,
    },
    {   // scalar vector mod 64
        seq_shoup_scalar_vector_mod_64,
        seq_shoup_scalar_vector_mod_64_vectorized,
        seq_shoup_scalar_vector_mod_64_unrolled,
        flint_shoup_scalar_vector_mod_64,
        avx2_shoup_scalar_vector_mod_64,
        avx2_shoup_scalar_vector_mod_64_unrolled,
#if defined(__AVX512F__)
        avx512_shoup_scalar_vector_mod_64,
        avx512_shoup_scalar_vector_mod_64_unrolled,
#endif
    },
    {   // dot prod 32
        seq_dot_product,
        seq_dot_product_vectorized,
        seq_dot_product_unrolled,
        split_dot_product_old,
        split_dot_product,
        split_kara_dot_product,
        simd2_dot_product,
        simd2_dot_product_unrolled,
#if defined(__AVX512F__)
        simd512_dot_product,
        simd512_dot_product_unrolled
#endif
    },
    {   // dot prod mod 32
        seq_dot_product_mod,
        seq_dot_product_mod_vectorized,
        seq_dot_product_mod_unrolled,
        flint_dot_product_mod,
        simd2_dot_product_mod,
        simd2_dot_product_mod_unrolled,
#if defined(__AVX512F__)
        simd512_dot_product_mod,
        simd512_dot_product_mod_unrolled,
#endif
    },
    {   // dot prod mod 64
        seq_dot_prod_mod_64,
        seq_dot_prod_mod_64_vectorized,
        seq_dot_prod_mod_64_unrolled,
        flint_dot_product_mod,
        split_dot_product_mod,
        split_kara_dot_product_mod,
        simd2_dot_prod_mod_64,
        simd2_split_dot_product_mod,
        simd2_kara_dot_product_mod,
#if defined(__AVX512F__)
        simd512_split_dot_product,
        simd512_kara_dot_product_mod,
#endif
    },
    {   // butterfly fft 64
        seq_fft,
        preinv_fft,
        preinv_split_fft,
        avx2_preinv_split_fft,
    },
    {   // lazy butterfly fft 64
        preinv_fft_lazy44,
        avx2_preinv_split_fft_lazy44,
#if defined(__AVX512F__)
        avx512_preinv_split_fft_lazy44,
#endif
    },
    };

    // number of versions for each function
    slong nbv[] = {5, 5, 5, 4, 6, 8, 6, 9, 4, 2};
#if defined(__AVX512F__)
    nbv[0] += 2; nbv[1] += 2; nbv[2] += 2; nbv[4] += 2; nbv[5] += 2; nbv[6] += 2; nbv[7] += 2; nbv[9] += 1;
#endif

    // name of the function
    const char* fnames[] = {
        "(64-bit) addition",
        "(64-bit) modular addition",
        "(32-bit) scalar-vector product",
        "(32-bit) modular scalar-vector product",
        "(64-bit) modular scalar-vector product",
        "(32-bit) dot product",
        "(32-bit) modular dot product",
        "(64-bit) modular dot product",
        "(64-bit) modular butterfly fft",
        "(64-bit) lazy butterfly fft (Harvey)",
    };

    // headers 
    char header2[][1024] = {
        "s-novec\t\ts-vec\t\ts-unr\t\tavx2\t\tavx2u",
        "s-novec\t\ts-vec\t\ts-unr\t\tavx2\t\tavx2u",
        "s-novec\t\ts-vec\t\ts-unr\t\tavx2\t\tavx2u",
        "s-novec\t\ts-vec\t\ts-unr\t\tavx2\n",
        "s-novec\t\ts-vec\t\ts-unr\t\tflint\t\tavx2\t\tavx2u",
        "s-novec\t\ts-vec\t\ts-unr\t\tsplit-o\t\tsplit-n\t\tkara\t\tavx2\t\tavx2u",
        "s-novec\t\ts-vec\t\ts-unr\t\tflint\t\tavx2\t\tavx2u",
        "s-novec\t\ts-vec\t\ts-unr\t\tflint\t\tsplit\t\tkara\t\tavx2\t\tsplit avx2\tkara avx2",
        "seq\t\tpreinv\t\tsplit\t\tavx2-split\n",
        "seq\t\tavx2",
    };


#if defined(__AVX512F__)
    strcat(header2[0], "\t\tavx512\t\tavx512u\n");
    strcat(header2[1], "\t\tavx512\t\tavx512u\n");
    strcat(header2[2], "\t\tavx512\t\tavx512u\n");

    strcat(header2[4], "\t\tavx512\t\tavx512u\n");
    strcat(header2[5], "\t\tavx512\t\tavx512u\n");
    strcat(header2[6], "\t\tavx512\t\tavx512u\n");
    strcat(header2[7], "\tsplit avx512\tkara avx512\n");

    strcat(header2[9], "\t\tavx512\n");
#else
    strcat(header2[0], "\n");
    strcat(header2[1], "\n");
    strcat(header2[2], "\n");

    strcat(header2[4], "\n");
    strcat(header2[5], "\n");
    strcat(header2[6], "\n");
    strcat(header2[7], "\n");

    strcat(header2[9], "\n");
#endif

    char header1[][1024] = {
        "seq no-vec | seq auto-vec | seq loop-unrolled | avx2 | avx2 loop-unrolled",
        "seq no-vec | seq auto-vec | seq loop-unrolled | avx2 | avx2 loop-unrolled",
        "seq no-vec | seq auto-vec | seq loop-unrolled | avx2 | avx2 loop-unrolled",
        "seq no-vec | seq auto-vec | seq loop-unrolled | avx2\n",
        "seq no-vec | seq auto-vec | seq loop-unrolled | flint | avx2 | avx2 loop-unrolled",
        "seq no-vec | seq auto-vec | seq loop-unrolled | split-old | split-new | karatsuba | avx2 | avx2 loop-unrolled",
        "seq no-vec | seq auto-vec | seq loop-unrolled | flint | avx2 | avx2 loop-unrolled",
        "seq no-vec | seq auto-vec | seq loop-unrolled | flint | seq split | seq karatsuba | avx2 | avx2 split | avx2 karatsuba",
        "seq | preinvert | split-preinvert | avx2 split-preinvert\n",
        "seq | avx2",
    };
#if defined(__AVX512F__)
    strcat(header1[0], " | avx512 | avx512 loop-unrolled\n");
    strcat(header1[1], " | avx512 | avx512 loop-unrolled\n");
    strcat(header1[2], " | avx512 | avx512 loop-unrolled\n");

    strcat(header1[4], " | avx512 | avx512 loop-unrolled\n");
    strcat(header1[5], " | avx512 | avx512 loop-unrolled\n");
    strcat(header1[6], " | avx512 | avx512 loop-unrolled\n");
    strcat(header1[7], " | avx512 split | avx512 karatsuba\n");

    strcat(header1[9], " | avx512\n");
#else
    strcat(header1[0], "\n");
    strcat(header1[1], "\n");
    strcat(header1[2], "\n");

    strcat(header1[4], "\n");
    strcat(header1[5], "\n");
    strcat(header1[6], "\n");
    strcat(header1[7], "\n");

    strcat(header1[9], "\n");
#endif


// PRINT PARAMETERS
    slong idfun;
    if (argc == 3)
    {
        info.bits = (flint_bitcnt_t)atoi(argv[1]);
        idfun = atoi(argv[2]);
    }
    else
    {
        flint_printf("ERROR: missing parameter(s).\n");
        flint_printf("Usage: ./profiler [bitsize] [idfunc]\n");
        flint_printf("  - bitsize: max number of bits for the entries or size of the modulo;\n");
        flint_printf("  - idfunc:\n");
        flint_printf("      #0 --> (64-bit) addition\n");
        flint_printf("      #1 --> (64-bit) modular addition\n");
        flint_printf("      #2 --> (32-bit) scalar-vector product\n");
        flint_printf("      #3 --> (32-bit) modular scalar-vector product\n");
        flint_printf("      #4 --> (64-bit) modular scalar-vector product\n");
        flint_printf("      #5 --> (32-bit) dot product\n");
        flint_printf("      #6 --> (32-bit) modular dot product\n");
        flint_printf("      #7 --> (64-bit) modular dot product\n");
        flint_printf("      #8 --> (64-bit) modular butterfly fft\n");
        flint_printf("      #9 --> (64-bit) lazy butterfly fft\n");
        return 0;
    }

    flint_printf("function: %s\n", fnames[idfun]);
    flint_printf("unit: all measurements in seconds\n");
    flint_printf("profiled: %s", header1[idfun]);
    flint_printf("bitsize: %ld\n\n", info.bits);
    flint_printf("len/func\t%s", header2[idfun]);



// BEGINNING OF PROFILING
    for (len = 10; len <= 200; len+=25)
    //for (len = 1; len < 200; ++len)
    {
        info.length = len;

        flint_printf("%d\t", len);
        for (slong v=0; v < nbv[idfun]; v++)
            funs[idfun](&info, versions[idfun][v]);
        flint_printf("\n");
    
    }

    for ( ; len <= 8000; len+=150)
    {
        info.length = len;

        flint_printf("%d\t", len);
        for (slong v=0; v < nbv[idfun]; v++)
            funs[idfun](&info, versions[idfun][v]);
        flint_printf("\n");
    }

    for (int i = 13; i < 23; i++)
    {
        info.length = 1 << i;

        flint_printf("%d\t", info.length);
        for (slong v=0; v < nbv[idfun]; v++)
            funs[idfun](&info, versions[idfun][v]);
        flint_printf("\n");

    }

    return 0;
}
