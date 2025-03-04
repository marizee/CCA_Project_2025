#ifndef SCALAR_VECTOR_32
#define SCALAR_VECTOR_32

#include <stdint.h>
#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"

void seq_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len) ;
void seq_scalar_vector_vectorized(nn_ptr res, ulong b, nn_ptr vec, slong len) ;
void seq_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len) ;

void simd2_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len) ;
void simd2_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len) ;

void simd512_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len) ;
void simd512_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len) ;

#endif