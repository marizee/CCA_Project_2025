#include "flint/profiler.h"
#include "flint/ulong_extras.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/flint.h"

#include "scalar_vector_32.h"
#include "scalar_vector_mod_32.h"
#include "dot_product_32.h"
#include "butterfly_fft_64.h"

#include <string.h>


typedef struct
{
   flint_bitcnt_t bits;
   slong length;
} info_t;

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
int main(int argc, char** argv)
{
    info_t info;
    slong len;

    typedef void (*func) ();
    typedef void (*timefun) (info_t*, func);
    const timefun funs[] = {scalar_vector, dot_prod, scalar_vector_mod, butterfly_fft_64};

    // all versions of the function
    const func versions[][10] = {
    {
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
    {
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
    {
        seq_mod_scalar_vector,
        seq_mod_scalar_vector_vectorized,
        seq_mod_scalar_vector_unrolled,
        simd2_mod_scalar_vector,
    },
    {
        seq_fft,
        preinv_fft,
        preinv_split_fft,
        avx2_preinv_split_fft,
    }
    };

    // number of versions for each function
    slong nbv[] = {5, 8, 4, 4};
#if defined(__AVX512F__)
    nbv[0] += 2; nbv[1] += 2;
#endif

    // name of the function
    const char* fnames[] = {"scalar-vector product", "dot product", "modular scalar-vector product", "modular butterfly fft"};

    // headers 
    char header2[][1024] = {
        "s-novec\t\ts-vec\t\ts-unr\t\tavx2\t\tavx2u",
        "s-novec\t\ts-vec\t\ts-unr\t\tsplit-o\t\tsplit-n\t\tkara\t\tavx2\t\tavx2u",
        "s-novec\t\ts-vec\t\ts-unr\t\tavx2\n",
        "seq\t\tpreinv\t\tsplit\t\tavx2-split\n",
    };
#if defined(__AVX512F__)
    strcat(header2[0], "\tavx512\t\tavx512u\n");
    strcat(header2[1], "\tavx512\t\tavx512u\n");
#else
    strcat(header2[0], "\n");
    strcat(header2[1], "\n");
#endif

    char header1[][1024] = {
        "seq no-vec | seq auto-vec | seq loop-unrolled | avx2 | avx2 loop-unrolled",
        "seq no-vec | seq auto-vec | seq loop-unrolled | split-old | split-new | karatsuba | avx2 | avx2 loop-unrolled",
        "seq no-vec | seq auto-vec | seq loop-unrolled | avx2",
        "seq | preinverse | split-preinverse | avx2 split-preinverse\n",
    };
#if defined(__AVX512F__)
    strcat(header1[0], " | avx512 | avx512 loop-unrolled\n");
    strcat(header1[1], " | avx512 | avx512 loop-unrolled\n");
    strcat(header1[2], " | avx512 | avx512 loop-unrolled\n");
#else
    strcat(header1[0], "\n");
    strcat(header1[1], "\n");
    strcat(header1[2], "\n");
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
        flint_printf("Usage: ./profile_sec2 [bitsize] [idfunc]\n");
        flint_printf("  - bitsize: max number of bits for the entries or size of the modulo;\n");
        flint_printf("  - idfunc:\n");
        flint_printf("      #0 --> scalar-vector product\n");
        flint_printf("      #1 --> dot product\n");
        flint_printf("      #2 --> modular scalar-vector product\n");
        flint_printf("      #3 --> modular butterfly fft\n");
        return 0;
    }
    
    flint_printf("function: %s\n", fnames[idfun]);
    flint_printf("unit: all measurements in seconds\n");
    flint_printf("profiled: %s", header1[idfun]);
    flint_printf("bitsize: %ld\n\n", info.bits);
    flint_printf("len/func\t%s", header2[idfun]);



// BEGINNING OF PROFILING
    for (len = 1; len < 200; ++len)
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
