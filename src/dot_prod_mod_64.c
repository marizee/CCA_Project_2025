#include "dot_prod_mod_64.h"
#include "mulsplit.h"

#include "flint/machine_vectors.h"

//#define MASK ((1L << 32) - 1)

// TODO : DELAYED REDUCTION IF POSSIBLE

void flint_dot_prod_mod_64(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    dot_params_t params = _nmod_vec_dot_params(len, mod);
    *res = _nmod_vec_dot(a, b, len, mod, params);
}

__attribute__((optimize("-fno-tree-vectorize")))
void seq_dot_prod_mod_64(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    ulong t_hi=0, t_mi=0, t_lo=0;
    ulong q_hi, q_lo;

    for (slong i = 0; i < len; i++)
    {
        umul_ppmm(q_hi, q_lo, a[i], b[i]);
        add_sssaaaaaa(t_hi, t_mi, t_lo, t_hi, t_mi, t_lo, 0, q_hi, q_lo);
    }
    NMOD_RED(t_hi, t_hi, mod);
    NMOD_RED3(*res, t_hi, t_mi, t_lo, mod);
}

void seq_dot_prod_mod_64_vectorized(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    ulong t_hi=0, t_mi=0, t_lo=0;
    ulong q_hi, q_lo;

    for (slong i = 0; i < len; i++)
    {
        umul_ppmm(q_hi, q_lo, a[i], b[i]);
        add_sssaaaaaa(t_hi, t_mi, t_lo, t_hi, t_mi, t_lo, 0, q_hi, q_lo);
    }
    NMOD_RED(t_hi, t_hi, mod);
    NMOD_RED3(*res, t_hi, t_mi, t_lo, mod);
}

void seq_dot_prod_mod_64_unrolled(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    ulong t_hi=0, t_mi=0, t_lo=0;
    ulong q_hi, q_lo;

    for (slong i = 0; i+3 < len; i+=4)
    {
        umul_ppmm(q_hi, q_lo, a[i+0], b[i+0]);
        add_sssaaaaaa(t_hi, t_mi, t_lo, t_hi, t_mi, t_lo, 0, q_hi, q_lo);
        umul_ppmm(q_hi, q_lo, a[i+1], b[i+1]);
        add_sssaaaaaa(t_hi, t_mi, t_lo, t_hi, t_mi, t_lo, 0, q_hi, q_lo);
        umul_ppmm(q_hi, q_lo, a[i+2], b[i+2]);
        add_sssaaaaaa(t_hi, t_mi, t_lo, t_hi, t_mi, t_lo, 0, q_hi, q_lo);
        umul_ppmm(q_hi, q_lo, a[i+3], b[i+3]);
        add_sssaaaaaa(t_hi, t_mi, t_lo, t_hi, t_mi, t_lo, 0, q_hi, q_lo);
    }
    NMOD_RED(t_hi, t_hi, mod);
    NMOD_RED3(*res, t_hi, t_mi, t_lo, mod);
}

void simd2_dot_prod_mod_64(ulong* res, nn_ptr a, nn_ptr b, slong len, nmod_t mod)
{
    //__m256i vt_hi = _mm256_setzero_si256();
    //__m256i vt_mi = _mm256_setzero_si256();
    //__m256i vt_lo = _mm256_setzero_si256();

    __m256i vq_hi = _mm256_setzero_si256();
    __m256i vq_lo = _mm256_setzero_si256();

    slong i;
    for (i = 0; i+3 < len; i += 4)
    {
        __m256i va = _mm256_loadu_si256((__m256i*)(a+i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(b+i));

        // umulppmm-avx2
        avx2_mul_split(&vq_hi, &vq_lo, va, vb);

        // TODO
    }
}
