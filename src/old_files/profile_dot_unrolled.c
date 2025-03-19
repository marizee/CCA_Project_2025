#include "flint/profiler.h"
#include "flint/ulong_extras.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/flint.h"

#include "dot_product_32_unrolled.h"

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

int main(int argc, char** argv)
{
    info_t info;
    slong len;

    typedef void (*func) ();
    typedef void (*timefun) (info_t*, func);
    const timefun funs[] = {scalar_vector, dot_prod, scalar_vector_mod};

    // all versions of the function
    const func versions[][10] = {
    {
    },
    {
        simd2_dot_product,
        simd2_dot_product_unrolled_8,
        simd2_dot_product_unrolled_16,
        simd2_dot_product_unrolled_32,
#if defined(__AVX512F__)
        simd512_dot_product,
        simd512_dot_product_unrolled
#endif
    },
    {
    }
    };

    // number of versions for each function
    slong nbv[] = {0, 4, 0};
#if defined(__AVX512F__)
    nbv[1] += 2;
#endif

    // name of the function
    const char* fnames[] = {"scalar-vector product", "dot product", "modular scalar-vector product"};

    // headers 
    char header2[][1024] = {
        "s-novec\t\ts-vec\t\ts-unr\t\tavx2\t\tavx2u",
        "avx2\t\tavx2-u2\t\tavx2-u4\t\tavx2-u8",
        "s-novec\t\ts-vec\t\ts-unr\t\tavx2\n",
    };
#if defined(__AVX512F__)
    strcat(header2[0], "\tavx512\t\tavx512u\n");
    strcat(header2[1], "\tavx512\t\tavx512u\n");
#else
    strcat(header2[0], "\n");
    strcat(header2[1], "\n");
#endif

    char header1[][1024] = {
        "",
        "avx2 (4) | avx2 loop-unrolled 2 (8) | avx2 loop-unrolled 4 (16) | avx2 loop-unrolled 8 (32)",
        "",
    };
#if defined(__AVX512F__)
    strcat(header1[1], " | avx512 | avx512 loop-unrolled\n");
#else
    strcat(header1[0], "\n");
    strcat(header1[1], "\n");
    strcat(header1[2], "\n");
#endif


// PRINT PARAMETERS
    slong idfun = 1;
    if (argc == 2)
    {
        info.bits = (flint_bitcnt_t)atoi(argv[1]);
    }
    else
    {
        flint_printf("ERROR: missing parameter(s).\n");
		flint_printf("Usage: %s [bitsize]\n", argv[0]);
        flint_printf("  - bitsize: max number of bits for the entries or size of the modulo;\n");
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
