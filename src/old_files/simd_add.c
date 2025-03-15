#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <inttypes.h>
#include <immintrin.h>

typedef uint32_t u32;
typedef uint64_t u64;
typedef long long i64;

// #define MOD_P

/**
 *  Compute the sum of two vectors of the same size.
 *  It assume their size n is a multiple of 4.
 *  If MOD_P is defined, the operation id done mod p, assuming a and b are
 *  already mod p.
 */
void simd_add(u32* a, u32* b, u32 n, u32 p, u32* res) {
	#ifdef MOD_P
	u64 tab[4] = { p, p, p, p };
	__m256i vp = _mm256_load_si256 ((const __m256i *)tab);
	__m256i vtrue = -_mm256_cmpeq_epi64(vp, vp);
	#endif

	for(u32 i = 0; i < n; i += 4) {
		u64 tab_a[4] = { a[i], a[i + 1], a[i + 2], a[i + 3] };
		__m256i va = _mm256_load_si256 ((const __m256i *)tab_a);

		u64 tab_b[4] = { b[i], b[i + 1], b[i + 2], b[i + 3] };
		__m256i vb = _mm256_load_si256 ((const __m256i *)tab_b);

		__m256i vres = _mm256_add_epi64 (va, vb);
		#ifdef MOD_P
		__m256i vmask = _mm256_cmpgt_epi64 (vp, vres);
		__m256i vres2 = _mm256_sub_epi64 (vres, vp);
		#endif

		u64 tmp[4];
		#ifndef MOD_P
		_mm256_store_si256 ((__m256i *)tmp, vres);
		#else
		_mm256_maskstore_epi64((i64 *)tmp, vmask, vres2);
		vmask = _mm256_andnot_si256(vmask, vtrue);
		_mm256_maskstore_epi64((i64 *)tmp, vmask, vres);
		#endif

		res[i + 0] = (u32) tmp[0];
		res[i + 1] = (u32) tmp[1];
		res[i + 2] = (u32) tmp[2];
		res[i + 3] = (u32) tmp[3];
	}
}
