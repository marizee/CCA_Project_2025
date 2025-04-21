#ifndef ADD_64
#define ADD_64

#include <stdint.h>
#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"

void seq_add(nn_ptr res, nn_ptr a, nn_ptr b, slong len) ;
void seq_add_vectorized(nn_ptr res, nn_ptr a, nn_ptr b, slong len) ;
void seq_add_unrolled(nn_ptr res, nn_ptr a, nn_ptr b, slong len) ;

void simd2_add(nn_ptr res, nn_ptr a, nn_ptr b, slong len) ;
void simd2_add_unrolled(nn_ptr res, nn_ptr a, nn_ptr b, slong len) ;

void simd512_add(nn_ptr res, nn_ptr a, nn_ptr b, slong len) ;
void simd512_add_unrolled(nn_ptr res, nn_ptr a, nn_ptr b, slong len) ;

#endif