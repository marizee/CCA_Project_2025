#ifndef BUTTERFLY_FFT
#define BUTTERFLY_FFT

void seq_fft(nn_ptr res_add, nn_ptr res_sub, nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod) ;
void preinv_fft(nn_ptr res_add, nn_ptr res_sub, nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod) ;
void preinv_fft_unrolled(nn_ptr res_add, nn_ptr res_sub, nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod) ;

void avx2_preinv_fft(nn_ptr res_add, nn_ptr res_sub, nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod) ;


#endif