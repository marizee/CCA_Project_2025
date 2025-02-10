# CCA_Project_2025


## Dump

- [Modular SIMD arithmetic in Mathemagix](https://arxiv.org/pdf/1407.3383)
- [Intel HEXL: Accelerating Homomorphic Encryption with Intel AVX512-IFMA52](https://arxiv.org/pdf/2103.16400)

- ~~[prefetch info](https://stackoverflow.com/questions/48994494/how-to-properly-use-prefetch-instructions)~~


## 07/02 - 14/02

### TODO

- [ ] install flint ppti
- [ ] test avx512
- [ ] improve profiling: 
    * 1 bitsize
    * 1 op/fichier 
    * columns = function
    * rows = vector size 1-200; 200-8000; millions
    * cycle/limb or timing %.3e

- [ ] implem other op

- [ ] start report



## 31/01 - 07/02

### TODO

- [x] timings with flint profiler (without modulo)

Seq:
- [X] use flint types
- [X] loop-unrolling for seq function
- [X] seq function without auto-vect

Intrinsics:
- [X] loop-unrolling for simd function
- [X] use avx512 in simd function with #if defined(__AVX512F__), #else, #endif
- ~~[ ] modulo in simd function~~
- [ ] test avx512 -> pb: flint not installed

Remark: machines CCA => only ppti-gpu-4 is usable (gpu-1: dossier etu inexistant, gpu-5: existe pas, gpu-3: pas avx512)

- machine 1 : Intel(R) Core(TM) Ultra 5 125H
- machine 2 : CPU xxx
- machine ppti: Intel® Xeon® Gold 6248 Processor  (Cascade Lake)

## General notes

### Compile time

- Compilation options: -O3 turns on all optimizations from -O2. (See: [Options that control optimization](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
Compile sequential function with `-fno-tree-vectorize`

- Loop-unrolling: can help the auto-vectorization...


### Understand timings

The number of cycles per limb of a function can be approximated by looking at the cpu clock speed:
X GHz => X billion cycles per second (See: [What is clock speed?](https://www.intel.com/content/www/us/en/gaming/resources/cpu-clock-speed.html).

Then we look at the throughput of the simd instructions we call, for a given cpu generation (See: [uops.info](https://uops.info/table.html).

Finally, we do a cross-multiplication with the number of operations performed by the measured function.

This does not take in account the latency or any other factor.

Ex: scalar-vector product of at most 32 bits integer with 4,5 GHz CPU and 16 384 elements
=> we except $\approx 3.64 \times 10^{-6}$

