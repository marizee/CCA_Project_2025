#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <time.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"
#include "flint/nmod.h"


void print_reg_64(char* nom, __m256i reg) 
{
    // prints values of the register assuming they are 64 bits integers.

    printf("%s =\t", nom);
    for (slong i=0; i<4; i++) {
        printf("%lld ", reg[i]);
    }
    printf("\n");
}

void print_reg_64d(char* nom, __m256d reg) 
{
    // prints values of the register assuming they are 64 bits integers.

    printf("%s =\t", nom);
    for (slong i=0; i<4; i++) {
        printf("%f ", reg[i]);
    }
    printf("\n");

}
void print_reg_64s(char* nom, __m256 reg) 
{
    // prints values of the register assuming they are 64 bits integers.

    printf("%s =\t", nom);
    for (slong i=0; i<4; i++) {
        printf("%f ", reg[i]);
    }
    printf("\n");
}


__attribute__((optimize("-fno-tree-vectorize")))
void seq_mod_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{
    // computes scalar-vector product of at most 32 bits integers.
    for (slong i=0; i < len; i++)
    {
        NMOD_RED(*(res+i), b*vec[i], mod);
    }
}

void seq_mod_scalar_vector_vectorized(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{
    // computes scalar-vector product of at most 32 bits integers with auto-vectorization.
    for (slong i=0; i < len; i++)
    {
        NMOD_RED(*(res+i), b*vec[i], mod);
    }
}

void seq_mod_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{
    // loop-unrolling of seq_scalar_vector.

    slong i;
    for (i=0; i+3 < len; i += 4)
    {
        NMOD_RED(res[i+0], b*vec[i+0], mod);
        NMOD_RED(res[i+1], b*vec[i+1], mod);
        NMOD_RED(res[i+2], b*vec[i+2], mod);
        NMOD_RED(res[i+3], b*vec[i+3], mod);
    }

    // when len is not a multiple of 4
    for ( ; i < len; i++)
    {
        NMOD_RED(res[i], b*vec[i], mod);
    }
}


// TODO: specify `-mno-avx512f` (uses avx2, but in case of)
void simd2_mod_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{
    // computes scalar-vector product of at most 32 bits integers using intrinsics.
    
    __m256i vb = _mm256_set1_epi64x((int)b);
    
    __m256i vmod = _mm256_set1_epi64x((int)mod.n);
    __m256 vinv = _mm256_set1_ps(1.0/mod.n);
    //print_reg_64s("vinv", vinv);

    slong i;
    for (i=0; i+3 < len; i+=4)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)&vec[i]);
        __m256i prod = _mm256_mul_epu32(vb, va);

        // compute floor(x/q)*q
        __m256 temp = _mm256_mul_ps(_mm256_cvtepi32_ps(prod), vinv);
        __m256i temp2 = _mm256_mul_epi32(_mm256_cvtps_epi32(temp), vmod);

        // temp2 - p to avoid negative results
        __m256i cmp = _mm256_cmpgt_epi32 (temp2, prod);
        __m256i and = _mm256_and_si256 (vmod, cmp);

        __m256i res2 = _mm256_sub_epi32(prod, _mm256_sub_epi32(temp2, and));
        _mm256_storeu_si256((__m256i *)&res[i], res2);
    }

    // when len is not a multiple of 4
    for ( ; i < len; i++)
    {
        NMOD_RED(res[i], b*vec[i], mod);
    }
}

/*
void simd2_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{

}

#if defined(__AVX512F__)
void simd512_mod_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{

}

void simd512_mod_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{

}
#endif
*/