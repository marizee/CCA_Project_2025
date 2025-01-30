#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <inttypes.h>
#include <immintrin.h>

void seq_vector_vector(uint64_t* b, uint64_t* a, uint32_t n, uint64_t* res) {
    for (uint32_t i=0; i<n; i++){
        res[i] = a[i]*b[i];
    }
}
