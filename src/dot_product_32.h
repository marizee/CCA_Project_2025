#ifndef DOT_PRODUCT_32
#define DOT_PRODUCT_32

#include <stdint.h>
#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"

void print_reg_64(char* nom, __m256i reg) ;

void seq_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
void seq_dot_product_vectorized(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
void seq_dot_product_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;

void split_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
void split_dot_product_old(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
void split_kara_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;

void simd2_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
void simd2_dot_product_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
void simd2_dot_product_unrolled_8(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
void simd2_dot_product_unrolled_16(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;

void simd512_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
void simd512_dot_product_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;

#endif
