#include "flint/profiler.h"
#include "flint/ulong_extras.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/flint.h"

#include "scalar_vector_32.h"


typedef struct
{
   flint_bitcnt_t bits;
   slong length;
} info_t;

void sample_no_vec(info_t* info)
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
    seq_scalar_vector(res,b,vec,length);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t\t%.3e", twall);
    
    _nmod_vec_clear(res);
    _nmod_vec_clear(vec);
    FLINT_TEST_CLEAR(state);
}

void sample_vec(info_t* info)
{
    // retrieve function parameters
    //info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;
    
    FLINT_TEST_INIT(state);
    
    //nmod_t mod;
    ulong n, b;
    nn_ptr vec, res;
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
    seq_scalar_vector_vectorized(res,b,vec,length);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);
    
    _nmod_vec_clear(res);
    _nmod_vec_clear(vec);
    FLINT_TEST_CLEAR(state);
}

void sample_unrolled(info_t* info)
{
    //info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;

    FLINT_TEST_INIT(state);
    
    //nmod_t mod;
    ulong n, b;
    nn_ptr vec, res;
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
    seq_scalar_vector_unrolled(res,b,vec,length);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);
    
   _nmod_vec_clear(vec);
   _nmod_vec_clear(res);
   FLINT_TEST_CLEAR(state);
}

void sample_simd2(info_t* info)
{
    //info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;

    FLINT_TEST_INIT(state);
    
    //nmod_t mod;
    ulong n, b;
    nn_ptr vec, res;
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
    simd2_scalar_vector(res,b,vec,length);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);
    
   _nmod_vec_clear(vec);
   _nmod_vec_clear(res);
   FLINT_TEST_CLEAR(state);
}

void sample_simd2_unrolled(info_t* info)
{
    //info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;

    FLINT_TEST_INIT(state);
    
    //nmod_t mod;
    ulong n, b;
    nn_ptr vec, res;
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
    simd2_scalar_vector_unrolled(res,b,vec,length);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);

   _nmod_vec_clear(vec);
   _nmod_vec_clear(res);
   FLINT_TEST_CLEAR(state);
}

#if defined(__AVX512F__)
void sample_simd512(info_t* info)
{
    //info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;

    FLINT_TEST_INIT(state);
    
    //nmod_t mod;
    ulong n, b;
    nn_ptr vec, res;
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
    simd512_scalar_vector(res,b,vec,length);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);
    
   _nmod_vec_clear(vec);
   _nmod_vec_clear(res);
   FLINT_TEST_CLEAR(state);
}

void sample_simd512_unrolled(info_t* info)
{
    //info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;

    FLINT_TEST_INIT(state);
    
    //nmod_t mod;
    ulong n, b;
    nn_ptr vec, res;
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
    simd512_scalar_vector_unrolled(res,b,vec,length);
    TIMEIT_STOP_VALUES(tcpu, twall)

    printf("\t%.3e", twall);
    
   _nmod_vec_clear(vec);
   _nmod_vec_clear(res);
   FLINT_TEST_CLEAR(state);
}
#endif

int main(int argc, char** argv)
{
    //// 200 + 52 + 10
    //double t[                 262]; // note: max seems to be consistently identical or extremely close to min
    //double mins_vec[             262];
    //double mins_unrolled[        262];
    //double mins_simd2[           262];
    //double mins_simd2_unrolled[  262];
    //#if defined(__AVX512F__)
    //double mins_simd512[         262];
    //double mins_simd512_unrolled[262];
    //#endif
    info_t info;
    slong len;
    //flint_bitcnt_t i;

    if (argc == 2)
    {
        info.bits = (flint_bitcnt_t)atoi(argv[1]);
    }
    else
    {
        flint_printf("ERROR: missing bitsize.\n");
        exit(0);
    }
    
    flint_printf("unit: all measurements in seconds\n");
    flint_printf("profiled: seq no-vec | seq auto-vec | seq loop-unrolled | avx2 | avx2 loop-unrolled");
#if defined(__AVX512F__)
    flint_printf(" | avx512 | avx512 loop-unrolled\n");
#else
    flint_printf("\n");
#endif
    flint_printf("bitsize: %ld\n\n", info.bits);
    
    flint_printf("len/func\t");
    flint_printf("s-novec\t\ts-autovec\ts-unr\t\tavx2\t\tavx2-unr");
#if defined(__AVX512F__)
    flint_printf("\tavx512\t\tavx512unr\n");
#else
    flint_printf("\n");
#endif


    for (len = 1; len < 200; ++len)
    {
        info.length = len;

        flint_printf("%d", len);
        sample_no_vec(          &info);
        sample_vec(             &info);
        sample_unrolled(        &info);
        sample_simd2(           &info);
        sample_simd2_unrolled(  &info);
#if defined(__AVX512F__)
        sample_simd512(         &info);
        sample_simd512_unrolled(&info);
#endif
        flint_printf("\n");
    
    }

    for ( ; len <= 8000; len+=150)
    {
        info.length = len;

        flint_printf("%d", len);
        sample_no_vec(          &info);
        sample_vec(             &info);
        sample_unrolled(        &info);
        sample_simd2(           &info);
        sample_simd2_unrolled(  &info);
#if defined(__AVX512F__)
        sample_simd512(         &info);
        sample_simd512_unrolled(&info);
#endif
        flint_printf("\n");
    
    }

    for (int i = 13; i < 23; i++)
    {
        info.length = 1 << i;

        flint_printf("%d", info.length);
        sample_no_vec(          &info);
        sample_vec(             &info);
        sample_unrolled(        &info);
        sample_simd2(           &info);
        sample_simd2_unrolled(  &info);
#if defined(__AVX512F__)
        sample_simd512(         &info);
        sample_simd512_unrolled(&info);
#endif
        flint_printf("\n");

    }

    return 0;
}
