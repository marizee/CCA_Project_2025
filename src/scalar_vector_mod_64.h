#ifndef SCALAR_VECTOR_MOD_64
#define SCALAR_VECTOR_MOD_64

#include <stdint.h>
#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"


void flint_shoup_scalar_vector_mod_64(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod) ;

void seq_shoup_scalar_vector_mod_64(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod) ;
void seq_shoup_scalar_vector_mod_64_vectorized(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod) ;
void seq_shoup_scalar_vector_mod_64_unrolled(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod) ;


void avx2_shoup_scalar_vector_mod_64(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod) ;
void avx2_shoup_scalar_vector_mod_64_unrolled(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod) ;

void avx512_shoup_scalar_vector_mod_64(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod) ;
void avx512_shoup_scalar_vector_mod_64_unrolled(nn_ptr res, ulong w, nn_ptr b, slong len, nmod_t mod) ;

#endif