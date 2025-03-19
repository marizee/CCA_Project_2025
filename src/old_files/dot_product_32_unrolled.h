#ifndef DOT_PRODUCT_32_UNR
#define DOT_PRODUCT_32_UNR

#include <stdint.h>
#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"


void simd2_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
void simd2_dot_product_unrolled_32(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
void simd2_dot_product_unrolled_16(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
void simd2_dot_product_unrolled_8(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;

#if defined(__AVX512F__)
void simd512_dot_product(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
void simd512_dot_product_unrolled(ulong* res, nn_ptr vec1, nn_ptr vec2, slong len) ;
#endif
#endif
