# CCA_Project_2025


## Dump

- [Modular SIMD arithmetic in Mathemagix](https://arxiv.org/pdf/1407.3383)
- [Intel HEXL: Accelerating Homomorphic Encryption with Intel AVX512-IFMA52](https://arxiv.org/pdf/2103.16400)

- [prefetch info](https://stackoverflow.com/questions/48994494/how-to-properly-use-prefetch-instructions)


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
- [ ] modulo in simd function
- [ ] test avx512 -> pb: flint not installed

Remark: machines CCA => only ppti-gpu-4 is usable (gpu-1: dossier etu inexistant, gpu-5: existe pas, gpu-3: pas avx512)

- machine 1 : CPU xxx
- machine 2 : CPU xxx
- machine ppti: Intel® Xeon® Gold 6248 Processor  (Cascade Lake)
