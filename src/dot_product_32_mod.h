#ifndef DOT_PRODUCT_MOD_32
#define DOT_PRODUCT_MOD_32

#include <stdint.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"


void flint_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;
void seq_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;
void seq_dot_product_mod_vectorized(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;

void split_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;
void split_kara_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;

void simd2_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;

void simd2_split_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;
void simd2_kara_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;

#if defined(__AVX512F__)
void simd512_split_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;
void simd512_kara_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;
#endif
#endif
