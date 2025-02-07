#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <time.h>

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

__attribute__((optimize("-fno-tree-vectorize")))
void seq_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{
    // computes sequential scalar-vector product modulo `mod` of at most 32 bits integers.
    for (slong i=0; i < len; i++){
        res[i] = (vec[i]*b);//%mod.n;
    }
}

void seq_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{
    // loop-unrolling of seq_scalar_vector.

    for (slong i=0; i+3 < len; i += 4)
    {
        res[i+0] = (b*vec[i+0]);//%mod.n;
        res[i+1] = (b*vec[i+1]);//%mod.n;
        res[i+2] = (b*vec[i+2]);//%mod.n;
        res[i+3] = (b*vec[i+3]);//%mod.n;
    }

    /* when len is not a multiple of 4 */
    for (slong i=(len-len%4); i < len; i++)
    {
        res[i] = (b*vec[i]);//%mod.n;
    }
}

void simd_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{
    // computes scalar-vector product of 32 bits integers using intrinsics.
    // TODO: modulo

    slong i;

#if defined(__AVX512F__)
    __m512i vb = _mm512_set1_epi64(b);

    for (i=0; i+7 < len; i+=8)
    {
        __m512i va = _mm512_loadu_si512((const __m512i *)&vec[i]);
        __m512i prod = _mm512_mul_epu32(vb, va);
        _mm512_storeu_si512 ((__m512i *)&res[i], prod);
    }

    // when n is not a multiple of 8
    for ( ; i < len; i++)
    {
        res[i] = b*vec[i];
    }
#else
    //int tmp = (int)b;
    //__m256i vb = _mm256_set_epi64x(tmp,tmp,tmp,tmp);
    __m256i vb = _mm256_set1_epi64x((int)b);

    for (i=0; i+3 < len; i+=4)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)&vec[i]);
        __m256i prod = _mm256_mul_epu32(va, vb);
        _mm256_storeu_si256((__m256i *)&res[i], prod);
    }

    // when n is not a multiple of 4
    for ( ; i < len; i++)
    {
        res[i] = b*vec[i];
    }
#endif
}

void simd_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod)
{
    // loop-unrolling of simd_scalar_vector.
    // TODO: modulo

    slong i;

#if defined(__AVX512F__)
    __m512i vb = _mm512_set1_epi64(b);

    for (i=0; i+31 < len; i+=32)
    {
        _mm512_storeu_si512((__m512i *)&res[i+ 0], _mm512_mul_epu32(vb, _mm512_loadu_si512((const __m512i *)&vec[i+ 0])));
        _mm512_storeu_si512((__m512i *)&res[i+ 8], _mm512_mul_epu32(vb, _mm512_loadu_si512((const __m512i *)&vec[i+ 8])));
        _mm512_storeu_si512((__m512i *)&res[i+16], _mm512_mul_epu32(vb, _mm512_loadu_si512((const __m512i *)&vec[i+16])));
        _mm512_storeu_si512((__m512i *)&res[i+24], _mm512_mul_epu32(vb, _mm512_loadu_si512((const __m512i *)&vec[i+24])));
    }

    // when n is not a multiple of 32
    for ( ; i < len; i++)
    {
        res[i] = vec[i]*b;
    }
#else
    //int tmp = (int)b;
    //__m256i vb = _mm256_set_epi64x(tmp,tmp,tmp,tmp);
    __m256i vb = _mm256_set1_epi64x((int)b);
    
    for (i=0; i+31 < len; i+=32)
    {
        _mm256_storeu_si256((__m256i *)&res[i+ 0], _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec[i+ 0]), vb));
        _mm256_storeu_si256((__m256i *)&res[i+ 4], _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec[i+ 4]), vb));
        _mm256_storeu_si256((__m256i *)&res[i+ 8], _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec[i+ 8]), vb));
        _mm256_storeu_si256((__m256i *)&res[i+12], _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec[i+12]), vb));
        _mm256_storeu_si256((__m256i *)&res[i+16], _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec[i+16]), vb));
        _mm256_storeu_si256((__m256i *)&res[i+20], _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec[i+20]), vb));
        _mm256_storeu_si256((__m256i *)&res[i+24], _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec[i+24]), vb));
        _mm256_storeu_si256((__m256i *)&res[i+28], _mm256_mul_epu32(_mm256_loadu_si256((const __m256i *)&vec[i+28]), vb));
    }
    
    // when n is not a multiple of 32
    for ( ; i < len; i++)
    {
        res[i] = b*vec[i];
    }
    
#endif
}
