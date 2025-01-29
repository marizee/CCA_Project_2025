#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <inttypes.h>
#include <immintrin.h>

typedef uint32_t u32;
typedef uint64_t u64;

int main() {
	const u64 n = 8;
	__attribute__ ((aligned (32))) u32 a[n]; // aligned 256 bits
	__attribute__ ((aligned (32))) u64 res[n]; // aligned 256 bits

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

	for(size_t i = 0; i < n ; i++) {
		a[i] = (u32) (1 << i);
		res[i] = (u32) 0;
		printf("%lu\t%u\n", i, a[i]);
	}

	for(size_t i = 0; i < n; i += 4) {
		__m256i tmp;
		tmp = _mm256_i32gather_epi64 ((void*)&a[i], vindex, sizeof(u32));
		tmp = _mm256_and_si256 (tmp, mask);
		_mm256_store_si256 ((__m256i *)&res[i], tmp);
		printf("i = %lu\n", i);
	}

	for(size_t j = 0; j < n; j++) {
		printf("%u\t%016lX\n", a[j], res[j]);
	}

	return 0;
}
