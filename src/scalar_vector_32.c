#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"


void print_reg_64(char* nom, __m256i reg) 
{
    // prints values of the register assuming they are 64 bits integers.

    printf("%s =\t", nom);
    for (slong i=0; i<4; i++) {
        printf("%lld ", reg[i]);
    }
    printf("\n");
}

__attribute__((optimize("-O2"))) void seq_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{
    // computes naive modular scalar-vector product of 32 bits integers.
    for (slong i=0; i < len; i++){
        res[i] = (vec[i]*b)%mod.n;
    }
}

void seq_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{
    // loop-unrolled version of seq_scalar_vector.

    for (slong i=0; i+3 < len; i += 4)
    {
        res[i+0] = (vec[i+0]*b)%mod.n;
        res[i+1] = (vec[i+1]*b)%mod.n;
        res[i+2] = (vec[i+2]*b)%mod.n;
        res[i+3] = (vec[i+3]*b)%mod.n;
    }

    /* when len is not a multiple of 4 */
    for (slong i=(len-len%4); i < len; i++)
    {
        res[i] = (vec[i]*b)%mod.n;
    }
}

void simd_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{
    // computes modular scalar-vector product of 32 bits integers using SIMD.

    // maybe use set (b,b,b,b,b,b)
    printf("b=%ld, int b=%d\n",b,(int)b);
    __m256i vb = _mm256_set1_epi64x((int)b);
    print_reg_64("vb", vb);

//#if __AVX512...__
//#else
//#endif
    for (slong i=0; i+3 < len; i+=4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)&vec[i]);
        __m256i prod = _mm256_mul_epu32(va, vb);
        print_reg_64("va", va);
        print_reg_64("prod", prod);

        //_mm256_store_si256((__m256i *)res[i], prod);
    }
//
    /* when n is not a multiple of 4 */
    //for (slong i=(len-len%4); i < len; i++)
    //{
    //    res[i] = vec[i]*b;
    //}
}