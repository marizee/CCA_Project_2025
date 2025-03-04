#ifndef SCALAR_VECTOR_MOD_32
#define SCALAR_VECTOR_MOD_32

#include <stdint.h>
#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"

void print_reg_64(char* nom, __m256i reg) ;

void seq_mod_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod) ;
void seq_mod_scalar_vector_vectorized(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod) ;
void seq_mod_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod) ;


void simd2_mod_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod) ;
void simd2_mod_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod) ;
/*
void simd512_scalar_vector(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod) ;
void simd512_scalar_vector_unrolled(nn_ptr res, ulong b, nn_ptr vec, slong len, nmod_t mod) ;
*/

#endif