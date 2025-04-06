#include "mulsplit.h"

#define MASK ((1L << 32) - 1)

// HIGH PART
//      a,b < 2**62
//      computes 64 bits high part of the product between a and b with split at 32
//      compared:
//      - umul_ppmm
//      - naive
//      - avx2
//      - avx512 mine

ulong mulhi_split(ulong a, ulong b)
{
    ulong r_hi, r_mi, r_lo;
    ulong a_lo, a_hi;
    ulong b_lo, b_hi;

    a_lo = a & MASK;
    a_hi = a >> 32;
    b_lo = b & MASK;
    b_hi = b >> 32;

    r_lo = a_lo*b_lo;
    r_hi = a_hi*b_hi;
    r_mi = a_lo*b_hi + a_hi*b_lo;

    // detects the carry if any
    ulong low = r_lo + (r_mi << 32);
    ulong carry = (low < r_lo ? 1 : 0);

    return (r_mi >> 32) + r_hi + carry;
}


// LOW PART
//      a,b < 2**62
//      computes low part of the product between a and b with split at 32
//      - umul_ppmm
//      - naive
//      - avx2 mine
//      - avx2 stack
//      - avx512 mullo

ulong mullo_split(ulong a, ulong b)
{
    ulong r_mi, r_lo;
    ulong a_lo, a_hi;
    ulong b_lo, b_hi;

    a_lo = a & MASK;
    a_hi = a >> 32;
    b_lo = b & MASK;
    b_hi = b >> 32;

    r_lo = a_lo*b_lo;
    r_mi = a_lo*b_hi + a_hi*b_lo;

    return r_lo + (r_mi << 32);
}
