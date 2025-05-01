#ifndef DOT_PROD_MOD_64
#define DOT_PROD_MOD_64

#include <stdint.h>
#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"

void flint_dot_prod_mod_64(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;

void seq_dot_prod_mod_64(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;
void seq_dot_prod_mod_64_vectorized(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;
void seq_dot_prod_mod_64_unrolled(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;

void simd2_dot_prod_mod_64(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;
void simd2_dot_prod_mod_unrolled(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;

#if defined(__AVX512F__)
void simd512_split_dot_product(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;
void simd512_kara_dot_product_mod(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;
#endif

#endif