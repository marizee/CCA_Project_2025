#include "flint/profiler.h"
#include "flint/ulong_extras.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/flint.h"

#include "scalar_vector_32.h"


#define MAX_BITS 31

typedef struct
{
   flint_bitcnt_t bits;
   slong length;
} info_t;

void sample(void * arg, ulong count)
{
    // retrieve function parameters OK
    info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;
    
    FLINT_TEST_INIT(state);
    
    nmod_t mod;
    ulong n, b;
    nn_ptr vec, res;
    ulong i;
    slong j;

    vec = _nmod_vec_init(length);
    res = _nmod_vec_init(length);

    for (i = 0; i < count; i++)
    {
        // init modulus
        n = n_randbits(state, (uint32_t)bits);
        if (n == UWORD(0)) n++;
        nmod_init(&mod, n);
        
        // init scalar and vector
        b = n_randint(state, n);
        for (j = 0; j < length; j++)
            vec[j] = n_randint(state, n);

        prof_start();
        for (slong j = 0; j < 100; j++)
            seq_scalar_vector(res,b,vec,length);
        prof_stop();
    }
    
    _nmod_vec_clear(res);
    _nmod_vec_clear(vec);
    FLINT_TEST_CLEAR(state);
}

void sample_unrolled(void * arg, ulong count)
{
    info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;

    FLINT_TEST_INIT(state);
    
    nmod_t mod;
    ulong n, b;
    nn_ptr vec, res;
    ulong i;
    slong j;

    vec = _nmod_vec_init(length);
    res = _nmod_vec_init(length);

    for (i = 0; i < count; i++)
    {
        // init modulus
        n = n_randbits(state, (uint32_t)bits);
        if (n == UWORD(0)) n++;
        nmod_init(&mod, n);
        
        // init scalar and vector
        b = n_randint(state, n);
        for (j = 0; j < length; j++)
            vec[j] = n_randint(state, n);
        
        prof_start();
        for (slong j = 0; j < 100; j++)
            seq_scalar_vector_unrolled(res,b,vec,length);
        prof_stop();
    }
    
   _nmod_vec_clear(vec);
   _nmod_vec_clear(res);
   FLINT_TEST_CLEAR(state);
}

void sample_simd(void * arg, ulong count)
{
    info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;

    FLINT_TEST_INIT(state);
    
    nmod_t mod;
    ulong n, b;
    nn_ptr vec, res;
    ulong i;
    slong j;

    vec = _nmod_vec_init(length);
    res = _nmod_vec_init(length);

    for (i = 0; i < count; i++)
    {
        // init modulus
        n = n_randbits(state, (uint32_t)bits);
        if (n == UWORD(0)) n++;
        nmod_init(&mod, n);
        
        // init scalar and vector
        b = n_randint(state, n);
        for (j = 0; j < length; j++)
            vec[j] = n_randint(state, n);
        
        prof_start();
        for (slong j = 0; j < 100; j++)
            simd2_scalar_vector(res,b,vec,length);
        prof_stop();
    }
    
   _nmod_vec_clear(vec);
   _nmod_vec_clear(res);
   FLINT_TEST_CLEAR(state);
}

void sample_simd_unrolled(void * arg, ulong count)
{
    info_t * info = (info_t *) arg;
    slong length = info->length;
    flint_bitcnt_t bits = info->bits;

    FLINT_TEST_INIT(state);
    
    nmod_t mod;
    ulong n, b;
    nn_ptr vec, res;
    ulong i;
    slong j;

    vec = _nmod_vec_init(length);
    res = _nmod_vec_init(length);

    for (i = 0; i < count; i++)
    {
        // init modulus
        n = n_randbits(state, (uint32_t)bits);
        if (n == UWORD(0)) n++;
        nmod_init(&mod, n);
        
        // init scalar and vector
        b = n_randint(state, n);
        for (j = 0; j < length; j++)
            vec[j] = n_randint(state, n);
        
        prof_start();
        for (slong j = 0; j < 100; j++)
            simd2_scalar_vector_unrolled(res,b,vec,length);
        prof_stop();
    }
    
   _nmod_vec_clear(vec);
   _nmod_vec_clear(res);
   FLINT_TEST_CLEAR(state);
}

int main(void)
{
    double min, max;
    double mins[18]; // note: max seems to be consistently identical or extremely close to min
    double mins_unrolled[18];
    double mins_simd[18];
    double mins_simd_unrolled[18];
    info_t info;
    flint_bitcnt_t i;

    flint_printf("unit: all measurements in c/l\n");
    flint_printf("profiled: sequential | seq loop-unrolled | simd | simd loop-unrolled\n");
    flint_printf("bit/len\t");

    for (int len = 1; len <= 16; ++len)
        flint_printf("%d\t\t", len);
    flint_printf("1024\t\t");
    flint_printf("65536\n");

    
    for (i = 2; i <= MAX_BITS; i++) // jusqua FLINT_BITS 
    {
        info.bits = i;

        // small sizes 
        for (int len = 1; len <= 16; ++len)
        {
            info.length = len;

            prof_repeat(&min, &max, sample, (void *) &info);
            mins[len-1] = min;
            prof_repeat(&min, &max, sample_unrolled, (void *) &info);
            mins_unrolled[len-1] = min;
            prof_repeat(&min, &max, sample_simd, (void *) &info);
            mins_simd[len-1] = min;
            prof_repeat(&min, &max, sample_simd_unrolled, (void *) &info);
            mins_simd_unrolled[len-1] = min;

        }
        
        info.length = 1024;
        prof_repeat(&min, &max, sample, (void *) &info);
        mins[16] = min;
        prof_repeat(&min, &max, sample_unrolled, (void *) &info);
        mins_unrolled[16] = min;
        prof_repeat(&min, &max, sample_simd, (void *) &info);
        mins_simd[16] = min;
        prof_repeat(&min, &max, sample_simd_unrolled, (void *) &info);
        mins_simd_unrolled[16] = min;

        info.length = 65536;
        prof_repeat(&min, &max, sample, (void *) &info);
        mins[17] = min;
        prof_repeat(&min, &max, sample_unrolled, (void *) &info);
        mins_unrolled[17] = min;
        prof_repeat(&min, &max, sample_simd, (void *) &info);
        mins_simd[17] = min;
        prof_repeat(&min, &max, sample_simd_unrolled, (void *) &info);
        mins_simd_unrolled[17] = min;
        

        if (i < FLINT_BITS)
        {
            flint_printf("%wd", i);
            for (int len = 1; len <= 16; ++len)
                flint_printf("\t%.1lf|%.1lf|%.1lf|%.1lf",
                            (mins[len-1]/(double)FLINT_CLOCK_SCALE_FACTOR)/(len*100),
                            (mins_unrolled[len-1]/(double)FLINT_CLOCK_SCALE_FACTOR)/(len*100),
                            (mins_simd[len-1]/(double)FLINT_CLOCK_SCALE_FACTOR)/(len*100),
                            (mins_simd_unrolled[len-1]/(double)FLINT_CLOCK_SCALE_FACTOR)/(len*100));
            flint_printf("\t%.1lf|%.1lf|%.1lf|%.1lf",
                    (mins[16]/(double)FLINT_CLOCK_SCALE_FACTOR)/(1024*100),
                    (mins_unrolled[16]/(double)FLINT_CLOCK_SCALE_FACTOR)/(1024*100),
                    (mins_simd[16]/(double)FLINT_CLOCK_SCALE_FACTOR)/(1024*100),
                    (mins_simd_unrolled[16]/(double)FLINT_CLOCK_SCALE_FACTOR)/(1024*100));
            flint_printf("\t%.1lf|%.1lf|%.1lf|%.1lf",
                    (mins[17]/(double)FLINT_CLOCK_SCALE_FACTOR)/(65536*100),
                    (mins_unrolled[17]/(double)FLINT_CLOCK_SCALE_FACTOR)/(65536*100),
                    (mins_simd[17]/(double)FLINT_CLOCK_SCALE_FACTOR)/(65536*100),
                    (mins_simd_unrolled[17]/(double)FLINT_CLOCK_SCALE_FACTOR)/(65536*100));
            flint_printf("\n");
        }/*
        else
        {
            flint_printf("%wd", i);
            for (int len = 1; len <= 16; ++len)
                flint_printf("\t%.1lf|%.1lf",
                            (mins[len-1]/(double)FLINT_CLOCK_SCALE_FACTOR)/(len*100),
                            (mins_unrolled[len-1]/(double)FLINT_CLOCK_SCALE_FACTOR)/(len*100));
            flint_printf("\t%.1lf|%.1lf\t",
                    (mins[16]/(double)FLINT_CLOCK_SCALE_FACTOR)/(1024*100),
                    (mins_unrolled[16]/(double)FLINT_CLOCK_SCALE_FACTOR)/(1024*100));
            flint_printf("\t%.1lf|%.1lf",
                    (mins[17]/(double)FLINT_CLOCK_SCALE_FACTOR)/(2048*100),
                    (mins_unrolled[17]/(double)FLINT_CLOCK_SCALE_FACTOR)/(2048*100));
            flint_printf("\n");
        }*/
    }

    return 0;
}
