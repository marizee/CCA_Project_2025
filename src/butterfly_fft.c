#include "butterfly_fft.h"

// unused at the moment
//#define SPLIT 20
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


void seq_fft(nn_ptr res_add, nn_ptr res_sub, nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod)
{
    // returns modular butterfly fft. w and coefficients of a and b are at most 32 bits.
    ulong prod;
    for (slong i=0; i<len; i++)
    {
        prod = nmod_mul(w, b[i], mod);
        res_add[i] = nmod_add(a[i], prod, mod);
        res_sub[i] = nmod_sub(a[i], prod, mod);
    }
}


void preinv_fft(nn_ptr res_add, nn_ptr res_sub, nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod)
{
    // returns modular butterfly fft with precomputation. 
    // w and coefficients of a and b are at most 32 bits.

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
        // NOTE: to think about: is umul_ppmm really necessary when b[i] and
        // w_pre are < 32 bits? (what is q_hi in this case?)

        // step2: r := b*w - q_hi*p
        res = w*b[i] - q_hi*mod.n; // no overflow (?!?!)
        // NOTE: there is overflow (e.g. w * b[i] is typically > 2**64, if the
        // input limit is ~60 bits), but formulas show (see FLINT's doc if you
        // want details) that this difference is in any case either the right
        // result or the right result + mod.n (and this thing is <= 64 bits, at
        // least if mod.n is <= 63 bits)

        // step3: (res >= p) ? res-p : res
        //NMOD_RED(res, res, mod);
        res = (res >= mod.n) ? res-mod.n : res;

        res_add[i] = nmod_add(a[i], res, mod);
        res_sub[i] = nmod_sub(a[i], res, mod);

    }
}

void preinv_fft_unrolled(nn_ptr res_add, nn_ptr res_sub, nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod)
{
    // returns modular butterfly fft with precomputation. 
    // w and coefficients of a and b are at most 32 bits.

    ulong w_pre = n_mulmod_precomp_shoup(w, mod.n);

    nn_ptr q_hi, q_lo;
    nn_ptr res;
    q_hi = _nmod_vec_init(len);
    q_lo = _nmod_vec_init(len);
    res = _nmod_vec_init(len);

    slong i;
    for (i=0; i+3<len; i+=4)
    {
        // flint
        //ulong r = n_mulmod_shoup(w, b[i], w_pre, mod.n);

        // handmade
        // step1: q_hi st b*w_pre = q_hi*2^32 + q_lo
        umul_ppmm(q_hi[i+0], q_lo[i+0], b[i+0], w_pre);
        umul_ppmm(q_hi[i+1], q_lo[i+1], b[i+1], w_pre);
        umul_ppmm(q_hi[i+2], q_lo[i+2], b[i+2], w_pre);
        umul_ppmm(q_hi[i+3], q_lo[i+3], b[i+3], w_pre);

        // step2: r := b*w - q_hi*p
        res[i+0] = w*b[i+0] - q_hi[i+0]*mod.n;
        res[i+1] = w*b[i+1] - q_hi[i+1]*mod.n;
        res[i+2] = w*b[i+2] - q_hi[i+2]*mod.n;
        res[i+3] = w*b[i+3] - q_hi[i+3]*mod.n;

        // step3: (res >= p) ? res-p : res
        //NMOD_RED(res, res, mod);
        res[i+0] = (res[i+0] >= mod.n) ? res[i+0]-mod.n : res[i+0];
        res[i+1] = (res[i+1] >= mod.n) ? res[i+1]-mod.n : res[i+1];
        res[i+2] = (res[i+2] >= mod.n) ? res[i+2]-mod.n : res[i+2];
        res[i+3] = (res[i+3] >= mod.n) ? res[i+3]-mod.n : res[i+3];

        res_add[i+0] = nmod_add(a[i+0], res[i+0], mod);
        res_add[i+1] = nmod_add(a[i+1], res[i+1], mod);
        res_add[i+2] = nmod_add(a[i+2], res[i+2], mod);
        res_add[i+3] = nmod_add(a[i+3], res[i+3], mod);

        res_sub[i+0] = nmod_sub(a[i+0], res[i+0], mod);
        res_sub[i+1] = nmod_sub(a[i+1], res[i+1], mod);
        res_sub[i+2] = nmod_sub(a[i+2], res[i+2], mod);
        res_sub[i+3] = nmod_sub(a[i+3], res[i+3], mod);
    }

    // len not multiple of 4
    for ( ; i<len; i++)
    {
        umul_ppmm(q_hi[i], q_lo[i], b[i], w_pre);
        res[i] = w*b[i] - q_hi[i]*mod.n; // no overflow (?!?!)
        res[i] = (res[i] >= mod.n) ? res[i]-mod.n : res[i];
        res_add[i] = nmod_add(a[i], res[i], mod);
        res_sub[i] = nmod_sub(a[i], res[i], mod);
    }
}


void avx2_preinv_fft(nn_ptr res_add, nn_ptr res_sub, nn_ptr a, nn_ptr b, ulong w, slong len, nmod_t mod)
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
        __m256i prod = _mm256_mul_epu32(vw_pre, vb);
        vq_hi = _mm256_srli_epi64(prod, 64);
        // TODO the 64 here is suspicious! what is the result of shifting by 64?
        // TODO understand why the result is still correct
        // --> this is related to the note above about q_hi when the inputs
        // are < 64 bits

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

        _mm256_storeu_si256((__m256i *)&res_add[i], add);
        _mm256_storeu_si256((__m256i *)&res_sub[i], sub);

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
        res[i] = w*b[i] - q_hi[i]*mod.n; // no overflow (?!?!)
        res[i] = (res[i] >= mod.n) ? res[i]-mod.n : res[i];
        res_add[i] = nmod_add(a[i], res[i], mod);
        res_sub[i] = nmod_sub(a[i], res[i], mod);
    }
}
