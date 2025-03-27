#ifndef DOT_PRODUCT_MOD_32
#define DOT_PRODUCT_MOD_32

#include <stdint.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"

void seq_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;
void seq_dot_product_mod_vectorized(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;
void seq_dot_product_mod_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;

void split_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;
void split_kara_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;

void simd2_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len, nmod_t mod) ;
void simd2_dot_product_mod_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;

#if defined(__AVX512F__)
void simd512_dot_product_mod(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
void simd512_dot_product_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
#endif
#endif