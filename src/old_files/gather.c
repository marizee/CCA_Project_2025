#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <inttypes.h>
#include <immintrin.h>

typedef uint32_t u32;
typedef uint64_t u64;

void simd_load(u32* a, u32 n, u64* res) {
	__m128i vindex;
	{
		__attribute__ ((aligned (32))) u32 tab[4] = { 0, 1, 2, 3 };
		vindex = _mm_load_si128((const __m128i *) tab);
	}

	__m256i mask; // create a mask to keep lower half of u64
	{
		__attribute__ ((aligned (32)))u64 tab[4] = { (u64) (u32) -1, (u64) (u32) -1, (u64) (u32) -1, (u64) (u32) -1 };
		mask = _mm256_load_si256 ((const __m256i *) tab);
	}

	u64 i;
	for(i = 0; i < n; i += 4) {
		__m256i tmp;
		tmp = _mm256_i32gather_epi64 ((void*)&a[i], vindex, sizeof(u32));
		tmp = _mm256_and_si256 (tmp, mask);
		_mm256_store_si256 ((__m256i *)&res[i], tmp);
	}
}

void naive_load(u32* a, u32 n, u64* res) {
	for(u32 i = 0; i < n; i += 4) {
		u64 tab[4] = { a[i], a[i + 1], a[i + 2], a[i + 3] };
		__m256i tmp = _mm256_load_si256((const __m256i *)tab);
		_mm256_store_si256 ((__m256i *)&res[i], tmp);
	}
}

int main() {
	const u32 n = 8;
	__attribute__ ((aligned (32))) u32 a[n]; // aligned 256 bits
	__attribute__ ((aligned (32))) u64 res[n]; // aligned 256 bits

	for(size_t i = 0; i < n ; i++) {
		a[i] = (u32) (1 << i);
		res[i] = (u32) 0;
		printf("%lu\t%u\n", i, a[i]);
	}

	simd_load(a, n, res);

	for(size_t j = 0; j < n; j++) {
		printf("%u\t%016lX\n", a[j], res[j]);
	}

	printf("n\tseq\tsimd\n");
	for (int i=1; i<17; i++) {
		uint32_t n = 1 << i;

		/* didn't manage to use 32 bits integers */
		__attribute__ ((aligned (32))) u32 a[n]; // aligned 256 bits
		__attribute__ ((aligned (32))) uint64_t res[n]; // aligned 256 bits

		uint32_t c = 0x8000; // 00001000 00000000 00000000 00000000
		for (uint32_t i=0; i<n; i++) {
			a[i] = (c-i);
		}

		clock_t start, end;
		double mean_seq = 0.0, mean_simd = 0.0;
		for (int k = 0; k<1000; k++) {
			start = clock();
			naive_load(a, n, res);
			end = clock();
			mean_seq += ((double) (end - start)) / CLOCKS_PER_SEC;

			start = clock();
			simd_load(a, n, res);
			end = clock();
			mean_simd += ((double) (end - start)) / CLOCKS_PER_SEC;
		}
		printf("%d\t%.4fs\t%.4fs\n", n, mean_seq, mean_simd);
	}

	return 0;
}
