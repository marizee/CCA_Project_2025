# CCA_Project_2025


## Dump

- [Modular SIMD arithmetic in Mathemagix](https://arxiv.org/pdf/1407.3383)
- [Intel HEXL: Accelerating Homomorphic Encryption with Intel AVX512-IFMA52](https://arxiv.org/pdf/2103.16400)

- ~~[prefetch info](https://stackoverflow.com/questions/48994494/how-to-properly-use-prefetch-instructions)~~

## Machines

- machine 1 : Intel(R) Core(TM) Ultra 5 125H  (Meteor Lake)
- machine 2 : CPU xxx
- machine ppti: Intel® Xeon® Gold 6248 Processor  (Cascade Lake, AVX512)
- machine argiope: AMD Ryzen 7 PRO 7840U  (Zen 4, AVX512)
- machine groebner: Intel(R) Xeon(R) Gold 6354  (IceLake, AVX512)

## TODO

### 14/02 - 04/03

- [ ] unroll +/- pour trouver le meilleur pas
- [X] verifier que avx2 est mesurée correctement sinon ajouter flag comme sur version sequentielle -> registres ymm (sinon zmm pour avx512). `-mno-avx512f` comment?
- [X] generaliser profiler pour mesurer autres fonctions

- [ ] ajouter le modulus: >= 32 bits (ex 45 bits)
    -> relire articles pour reduction

dot product:
- faire des versions split_k (avec k constante C) par ex 20 bits -> pas concluant.
    - [X] style produit normal 
        r = (lo1 * lo2) + 2^k(lo1 * hi2 + hi1 * lo2) + 2^{2k}(hi1 * hi2)
    - [X] style karatsuba 
        rlo = lo1 * lo2;
        rhi = hi1 * hi2;
        rmid = (lo1 + hi1) * (lo2 + hi2) - rlo - rhi;
        r = rlo + 2^k(rmid) + 2^{2k}(rhi)

Remarques:
- split26 -> jusqu'à len=1000
- horizontal sum -> utiliser fonctions de flint (voir fichier Vincent)


butterfly fft: (See: https://flintlib.org/doc/ulong_extras.html ).
- voir tableau
- commencer avec une seule paire de coeff

___
### 07/02 - 14/02

- [X] install flint ppti
- [X] test avx512
- [X] improve profiling: 
    * 1 bitsize
    * 1 op/fichier 
    * columns = function
    * rows = vector size 1-200; 200-8000; millions
    * cycle/limb or timing %.3e

- implem other op
    - [X] dot product -> TODO: fix overflow (See: https://github.com/vneiger/pml/blob/61383d9ae20853fef2179ca585291bb8da74c2fc/flint-extras/nmod_vec_extra/src/nmod_vec_dot_product.c#L205C7-L205C38 ).
    - [ ] butterfly fft
    - ~~[ ] vector-vector product~~

- [ ] start report

#### Notes RDV

Pour des multiples de 8, l'auto vec se compare à l'avx512.

Pour scalar-vector, ppti a un meilleur facteur. S'explique par les throughput: ppti 0.5/cycle - groebner 1/cycle - vincent 0.5 à 1/cycle.
=> avx512 pas encore stable mais en cours d'améliorations (comparer avec vieux proc).


___
### 31/01 - 07/02

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


## General notes

### Compile time

- Compilation options: -O3 turns on all optimizations from -O2. (See: [Options that control optimization](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
Compile sequential function with `-fno-tree-vectorize`

- Loop-unrolling: can help the auto-vectorization...


### Understand timings

The number of cycles per limb of a function can be approximated by looking at the cpu clock speed:
X GHz => X billion cycles per second (See: [What is clock speed?](https://www.intel.com/content/www/us/en/gaming/resources/cpu-clock-speed.html ).

Then we look at the throughput of the simd instructions we call, for a given cpu generation (See: [uops.info](https://uops.info/table.html ).

Finally, we do a cross-multiplication with the number of operations performed by the measured function.

This does not take in account the latency or any other factor.

Ex: scalar-vector product of at most 32 bits integer with 4,5 GHz CPU and 16 384 elements
=> we except $\approx 3.64 \times 10^{-6}$

