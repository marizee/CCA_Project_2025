#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <inttypes.h>
#include <immintrin.h>
//#include <x86intrin.h>          // works
     

void print_reg_32(char* nom, __m256i reg) {
    uint32_t* r = (uint32_t *)&reg;
    printf("%s =\t", nom);
    for (uint32_t i=0; i<8; i++) {
        printf("%d ", r[i]);
    }
    printf("\n");
}

void print_reg_64(char* nom, __m256i reg) {
    uint64_t* r = (uint64_t *)&reg;
    printf("%s =\t", nom);
    for (uint32_t i=0; i<4; i++) {
        printf("%ld ", r[i]);
    }
    printf("\n");
}

void seq_vector_vector(uint64_t* b, uint64_t* a, uint32_t n, uint64_t* res) {
    for (uint32_t i=0; i<n; i++){
        res[i] = a[i]*b[i];
    }
}

void simd_vector_vector(uint64_t* b, uint64_t* a, uint32_t n, uint64_t* res) {

    for (uint32_t i=0; i<n; i+=4) {
        __m256i va = _mm256_load_si256((const __m256i *)&a[i]);
        __m256i vb = _mm256_load_si256((const __m256i *)&b[i]);

        /* Multiply the low unsigned 32-bit integers from each packed 64-bit
        element in a and b, and store the unsigned 64-bit results in dst */
        __m256i prod = _mm256_mul_epu32(va, vb);

        _mm256_store_si256((__m256i *)&res[i], prod);
    }

    /* when n is not a multiple of 4 */
    for (uint32_t i=(n-n%4); i<n; i++) {
        res[i] = a[i]*b[i];
    }
}


int main(int arc, char** argv) {

    printf("n\tseq\tsimd\n");
    for (int i=1; i<17; i++) {
        uint32_t n = 1 << i;

        /* didn't manage to use 32 bits integers */
        __attribute__ ((aligned (32))) uint64_t a[n]; // aligned 256 bits
        __attribute__ ((aligned (32))) uint64_t b[n]; // aligned 256 bits
        __attribute__ ((aligned (32))) uint64_t res[n]; // aligned 256 bits

        uint32_t c = 0x8000; //1000 00000000 00000000 
        for (uint32_t i=0; i<n; i++) {
            a[i] = (c-i);
            b[i] = (c-i);
        }

        clock_t start, end;
        double mean_seq = 0.0, mean_simd = 0.0;
        for (int k = 0; k<1000; k++) {
            start = clock();
            seq_vector_vector(b,a,n,res);
            end = clock();
            mean_seq += ((double) (end - start)) / CLOCKS_PER_SEC;

            //printf("Timing: %.5fs\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);
            //printf("res=\t");
            //for (uint32_t i=0; i<n; i++) { printf("%ld ", res[i]);}
            //printf("\n\n");

            start = clock();
            simd_vector_vector(b,a,n,res);
            end = clock();
            mean_simd += ((double) (end - start)) / CLOCKS_PER_SEC;

            //printf("Timing: %.5fs\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);
            //printf("res=\t");
            //for (uint32_t i=0; i<n; i++) { printf("%ld ", res[i]);}
            //printf("\n");
        }
        printf("%d\t%.4fs\t%.4fs\n", n, mean_seq, mean_simd);
    }
    return 0;
}