#include "butterfly_fft_32.h"




ulong n_mulmod_precomp_shoup_32(ulong w, ulong n)
{
    // returns w_precomp = floor(w*2**32 /n)

    return (w << 32)/n;
}

void seq_fft_32(nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod)
{
    // returns modular butterfly fft. 
    // w and coefficients of a and b are already reduced mod n.
    
    ulong prod;
    for (slong i=0; i<len; i++)
    {
        prod = nmod_mul(w, b[i], mod);
        b[i] = nmod_sub(a[i], prod, mod);
        a[i] = nmod_add(a[i], prod, mod);
    }
}


void preinv_fft_32(nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod)
{
    // returns modular butterfly fft with precomputation. 
    // w and coefficients of a and b are  already reduced mod n.

    ulong w_pre = n_mulmod_precomp_shoup_32(w, mod.n);
    //printf("w_pre=%ld\n", w_pre);

    ulong q_hi, res;
    
    for (slong i=0; i<len; i++)
    {
        q_hi = (w_pre*b[i]) >> 32;
        res = w*b[i] - q_hi*mod.n;

        res = (res >= mod.n) ? res-mod.n : res;
        
        b[i] = nmod_sub(a[i], res, mod);
        a[i] = nmod_add(a[i], res, mod);
    }
}



void avx2_preinv_fft_32(nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod)
{

    ulong w_pre = n_mulmod_precomp_shoup_32(w, mod.n);

    __m256i vw = _mm256_set1_epi64x(w);
    __m256i vw_pre = _mm256_set1_epi64x(w_pre);

    __m256i vq_hi = _mm256_setzero_si256();
    __m256i vres = _mm256_setzero_si256();

    __m256i vmod = _mm256_set1_epi64x(mod.n);
    __m256i vzero = _mm256_setzero_si256();


    slong i;
    for (i=0; i+3<len; i+=4)
    {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a+i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b+i));

        // step1: q_hi st b*w_pre = q_hi*2^32 + q_lo
        __m256i prod = _mm256_mul_epu32(vw_pre, vb);
        vq_hi = _mm256_srli_epi64(prod, 32);

        // step2: r := b*w - q_hi*p
        vres = _mm256_sub_epi64(_mm256_mul_epu32(vw, vb), _mm256_mul_epu32(vq_hi, vmod));

        // step3: (res >= p) ? res-p : res
        __m256i cmp = _mm256_cmpgt_epi64(vres, vmod);
        __m256i and = _mm256_and_si256 (vmod, cmp);
        vres = _mm256_sub_epi64(vres, and);

        __m256i add = _mm256_add_epi64(va, vres);
        __m256i sub = _mm256_sub_epi64(va, vres);

        // modular reduction
        __m256i cmp_add = _mm256_cmpgt_epi64(add, vmod);
        __m256i add_mask = _mm256_and_si256 (cmp_add, vmod); // set to vmod if add is greater than mod; 0 else
        add = _mm256_sub_epi64(add, add_mask);

        __m256i cmp_sub = _mm256_cmpgt_epi64(sub, vzero);
        __m256i sub_mask = _mm256_andnot_si256 (cmp_sub, vmod); // set to vmod if sub is negative; 0 else
        sub = _mm256_add_epi64(sub, sub_mask);

        _mm256_storeu_si256((__m256i *)&a[i], add);
        _mm256_storeu_si256((__m256i *)&b[i], sub);

    }

    // len not multiple of 4
    nn_ptr q_hi, q_lo;
    nn_ptr res;
    q_hi = _nmod_vec_init(len);
    q_lo = _nmod_vec_init(len);
    res = _nmod_vec_init(len);

    for ( ; i<len; i++)
    {
        umul_ppmm(q_hi[i], q_lo[i], b[i], w_pre);
        res[i] = w*b[i] - q_hi[i]*mod.n;
        res[i] = (res[i] >= mod.n) ? res[i]-mod.n : res[i];
        b[i] = nmod_sub(a[i], res[i], mod);
        a[i] = nmod_add(a[i], res[i], mod);
    }
}