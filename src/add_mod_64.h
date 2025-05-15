#ifndef ADD_MOD_64
#define ADD_MOD_64

#include <stdint.h>
#include <immintrin.h>

#include "flint/flint.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"

void flint_add_mod(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;

void seq_add_mod(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;
void seq_add_mod_vectorized(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;
void seq_add_mod_unrolled(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;

void simd2_add_mod(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;
void simd2_add_mod_unrolled(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;

void simd512_add_mod(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;
void simd512_add_mod_unrolled(nn_ptr res, nn_ptr a, nn_ptr b, slong len, nmod_t mod) ;

#endif