#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <inttypes.h>
#include <immintrin.h>

typedef uint32_t u32;
typedef uint64_t u64;

void naive_load(u32* a, u32 n, u64* res) {
	for(u32 i = 0; i < n; i += 4) {
		u64 tab[4] = { a[i], a[i + 1], a[i + 2], a[i + 3] };
		__m256i tmp = _mm256_load_si256((const __m256i *)tab);
		_mm256_store_si256 ((__m256i *)&res[i], tmp);
	}
}
