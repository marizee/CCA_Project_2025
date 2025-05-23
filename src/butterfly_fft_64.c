#include "butterfly_fft_64.h"
#include "mulsplit.h"

#define SPLIT 32
//#define MASK ((1L << SPLIT) - 1)

// FIXME better to do two versions: one assuming <= 32 bits, another one
// assuming <= 64 bits (or maybe 30 and 62, I will explain when we meet)
// --> note that seq_fft below is already correct up to 64 bits
// --> note that Shoup's multiplication works as soon as <= 63 bits (cf doc),
// so preinv_fft below is correct up to 63 bits
//
// for the 32 bits version, we actually don't need umul_ppmm (think about it)
// and we can hope that automatic vectorization will work not so bad (and, at
// least, we can exploit handwritten avx2 with things like _mm256_mul_epu32)
//
// for the 64 bits version, things will be trickier, we need to find a way to
// circumvent the umul_ppmm, e.g. with some splitting into 32 bit low part and
// 32 bit high part
// 
// both versions are interesting!


void seq_fft(nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod)
{
    // returns modular butterfly fft over at most 64 bits integers.
    // w and coefficients of a and b are already reduced mod n.
    
    ulong prod;
    for (slong i=0; i<len; i++)
    {
        prod = nmod_mul(w, b[i], mod);
        b[i] = nmod_sub(a[i], prod, mod);
        a[i] = nmod_add(a[i], prod, mod);
    }
}

void preinv_fft(nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod)
{
    // returns modular butterfly fft with precomputation over at most 64 bits integers. 
    // w and coefficients of a and b are  already reduced mod n.

    ulong w_pre = n_mulmod_precomp_shoup(w, mod.n);

    ulong q_hi, q_lo;
    ulong res;
    
    for (slong i=0; i<len; i++)
    {
        // flint
        //ulong r = n_mulmod_shoup(w, b[i], w_pre, mod.n);

        // handmade
        // step1: q_hi st b*w_pre = q_hi*2^64 + q_lo
        umul_ppmm(q_hi, q_lo, b[i], w_pre);

        // step2: r := b*w - q_hi*p
        res = w*b[i] - q_hi*mod.n;
        // NOTE: there is overflow (e.g. w * b[i] is typically > 2**64, if the
        // input limit is ~60 bits), but formulas show (see FLINT's doc if you
        // want details) that this difference is in any case either the right
        // result or the right result + mod.n (and this thing is <= 64 bits, at
        // least if mod.n is <= 63 bits)

        // step3: (res >= p) ? res-p : res
        res = (res >= mod.n) ? res-mod.n : res;

        b[i] = nmod_sub(a[i], res, mod);
        a[i] = nmod_add(a[i], res, mod);
    }
}

void preinv_split_fft(nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod)
{
    // returns modular butterfly fft with precomputation and split over at most 64 bits integers. 
    // w and coefficients of a and b are  already reduced mod n.

    ulong w_pre = n_mulmod_precomp_shoup(w, mod.n);

    ulong q_hi=0;
    ulong res;
    
    for (slong i=0; i<len; i++)
    {
        // step1: q_hi st b*w_pre = q_hi*2^64 + q_lo
        //umul_ppmm(q_hi, q_lo, b[i], w_pre);
        q_hi = mulhi_split(b[i], w_pre);

        // step2: r := b*w - q_hi*p
        res = w*b[i] - q_hi*mod.n;

        // step3: (res >= p) ? res-p : res
        res = (res >= mod.n) ? res-mod.n : res;
        // (line above not needed in total lazy)

        b[i] = nmod_sub(a[i], res, mod);
        a[i] = nmod_add(a[i], res, mod);
        // total lazy:
        //b[i] = a[i] - res;
        //a[i] = a[i] + res;
    }
}

void avx2_preinv_split_fft(nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod)
{
    ulong w_pre = n_mulmod_precomp_shoup(w, mod.n);

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
        vq_hi = avx2_mulhi_split_v2(vw_pre, vb); // OK

        // step2: r := b*w - q_hi*p OK
        __m256i llo = avx2_mullo_epi64(vw, vb);
        __m256i rlo = avx2_mullo_epi64(vq_hi, vmod);
        vres = _mm256_sub_epi64(llo, rlo); // only low part is needed

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
        __m256i sub_mask = _mm256_andnot_si256(cmp_sub, vmod); // set to vmod if sub is negative; 0 else
        sub = _mm256_add_epi64(sub, sub_mask);

        _mm256_storeu_si256((__m256i *)&a[i], add);
        _mm256_storeu_si256((__m256i *)&b[i], sub);
    }

    ulong q_hi=0, q_lo=0;
    ulong res;
    
    for ( ; i<len; i++)
    {
        // step1: q_hi st b*w_pre = q_hi*2^64 + q_lo
        umul_ppmm(q_hi, q_lo, b[i], w_pre);

        // step2: r := b*w - q_hi*p
        res = w*b[i] - q_hi*mod.n;

        // step3: (res >= p) ? res-p : res
        res = (res >= mod.n) ? res-mod.n : res;

        b[i] = nmod_sub(a[i], res, mod);
        a[i] = nmod_add(a[i], res, mod);
    }
}
