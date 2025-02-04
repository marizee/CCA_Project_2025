# CCA_Project_2025


## 31/01 - 07/02

### TODO

- [ ] timings with flint profiler (pb with struct `flint_rand_t state` for random)

Seq:
- [X] use flint types
- [X] loop-unrolling for seq function
- [X] seq function without auto-vect

Intrinsics:
- [ ] loop-unrolling for simd function (how??)
- [X] use avx512 in simd function with #if defined(__AVX512F__), #else, #endif
- [ ] modulo in simd function
- [ ] test avx512 -> pb: flint not installed

Remark: machines CCA => only ppti-gpu-4 is usable (gpu-1: dossier etu inexistant, gpu-5: existe pas, gpu-3: pas avx512)
