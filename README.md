# PCCA 2025

Implementation of modular integer arithmetic operations using the SIMD vectorization provided by
the Intel Advanced Vector Extensions.

## Usage

Build all the executables:
```
$ make all
```

Run the profiling of an operation:
```
$ ./profiler [bitsize] [idfunc]
```
where
- `bitsize`: maximum size in bits for the entries or size of the modulus;
- `idfunc`:
    - #0 --> (64-bit) addition
    - #1 --> (64-bit) modular addition
    - #2 --> (32-bit) scalar-vector product
    - #3 --> (32-bit) modular scalar-vector product
    - #4 --> (64-bit) modular scalar-vector product
    - #5 --> (32-bit) dot product
    - #6 --> (32-bit) modular dot product
    - #7 --> (64-bit) modular dot product
    - #8 --> (64-bit) modular butterfly fft
    - #9 --> (64-bit) lazy butterfly fft

## Authors
* **ASSIRE Damien - 21112838**
* **BONBOIRE Marie - 21100552**