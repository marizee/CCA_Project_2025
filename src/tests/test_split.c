#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>

#include <stdbool.h>

#include "flint/flint.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"
#include "flint/profiler.h"

/* Notes
    - the value of splitbits corresponds to the size of the low part, the size of the high part is bitsize-splitbits
*/

//#define SPLIT_BITS 20   // < 32
//#define BIT_SIZE 32

void split(ulong* res, nn_ptr a, nn_ptr b, ulong split, ulong mask, slong len){

    ulong alo, ahi, blo, bhi;
    ulong rlo=0, rmid=0, rhi=0;
    
    for (slong i=0; i < len; i++)
    {
        bhi = b[i] >> split;
        ahi = a[i] >> split;
        alo = a[i] & mask;
        blo = b[i] & mask;

    //printf("\talo=%ld; ahi=%ld\n", alo, ahi);
    //printf("\tblo=%ld; bhi=%ld\n", blo, bhi);

        rlo += alo*blo;
        rhi += ahi*bhi;
        rmid += alo*bhi + ahi*blo;
    }
    //printf("\trlo=%ld; rmid=%ld; rhi=%ld\n", rlo, rmid, rhi);

    *res = rlo + (rmid << split) + (rhi << 2*split);
}

void split_kara(ulong* res, nn_ptr a, nn_ptr b, ulong split, ulong mask, slong len){

    ulong alo, ahi, blo, bhi;
    ulong lolo, hihi;
    ulong rlo=0, rmid=0, rhi=0;
    
    for (slong i=0; i < len; i++)
    {
        bhi = b[i] >> split;
        ahi = a[i] >> split;
        alo = a[i] & mask;
        blo = b[i] & mask;

        //printf("\talo=%ld; ahi=%ld\n", alo, ahi);
        //printf("\tblo=%ld; bhi=%ld\n", blo, bhi);


        lolo = alo*blo;
        hihi = ahi*bhi;

        rlo += lolo;
        rhi += hihi;
        rmid += (alo + ahi)*(blo + bhi) - lolo - hihi;
        //printf("\trlo=%ld; rmid=%ld; rhi=%ld\n", rlo, rmid, rhi);

    }

    *res = rlo + (rmid << split) + (rhi << 2*split);
}


int main() {

    slong len = 1 << 20;
    //flint_bitcnt_t bits = BIT_SIZE;
    FLINT_TEST_INIT(state);

    //nmod_t mod;
    ulong n;
    nn_ptr a, b;

    clock_t start, end;
    double tseq, tsplit1, tsplitk;

    // seq timings
    double t[28];

    // min timing for each bit size & its associated value of split
    double mins[28];
    slong mins_k[28];

    double mins_kara[28];
    slong mins_k_kara[28];

    for (slong i=0; i < 28; i++)
    {
        mins[i] = 10.; 
        mins_kara[i] = 10.; 
    }

    printf("bits\tsplit\tcheck\ttseq\ttsplit\ttkara\n");
    for (flint_bitcnt_t bits = 4; bits < 32; bits++)
    {
        // init modulus structure
        n = n_randbits(state, (uint32_t)bits);
        if (n == UWORD(0)) n++;
        //nmod_init(&mod, n);

        a = _nmod_vec_init(len);
        b = _nmod_vec_init(len);
        for (slong i = 0; i < len; i++)
        {
            a[i] = n_randint(state, n);
            b[i] = n_randint(state, n);
        }
    
        //_nmod_vec_print_pretty(a, len, mod);
        //_nmod_vec_print_pretty(b, len, mod);
        //flint_printf("a=%ld; b=%ld\n", a, b);

        ulong res1=0, res2=0, res3=0;

        start = clock();
        for (slong i = 0; i < len; i++)
            res1 += a[i]*b[i];
        end = clock();
        tseq = ((double) (end - start)) / CLOCKS_PER_SEC;
        t[bits-4] = tseq;

        //flint_printf("\t\t\tres1=%ld\n", res1);

        //ulong tmp = (3*bits/4+1 < 32) ? 3*bits/4+1 : 32;
        ulong tmp = bits;
        //printf("tmp=%ld\n",tmp);

 

        for (ulong k = 1; k < tmp; k ++) //3*bits/4
        {
            ulong mask = ((1L << k) - 1);
            start = clock();
            split(&res2, a, b, k, mask, len);
            end = clock();
            tsplit1 = ((double) (end - start)) / CLOCKS_PER_SEC;
            //flint_printf("\t\t\tres2=%ld\n", res2);
            if (tsplit1 < mins[bits-4])
            {
                mins[bits-4] = tsplit1;
                mins_k[bits-4] = k; 
            }

            start = clock();
            split_kara(&res3, a, b, k, mask, len);
            end = clock();
            tsplitk = ((double) (end - start)) / CLOCKS_PER_SEC;
            //flint_printf("\t\t\tres3=%ld\n", res3);
            if (tsplit1 < mins_kara[bits-4])
            {
                mins_kara[bits-4] = tsplitk;
                mins_k_kara[bits-4] = k; 
            }


            int check1 = (res1 == res2);
            int check2 = (res2 == res3);

            printf("%ld\t%ld\t%d\t%.5f\t%.5f\t%.5f\n", bits, k, check1 & check2, tseq, tsplit1, tsplitk);

        }
        _nmod_vec_clear(a);
        _nmod_vec_clear(b);
        FLINT_TEST_CLEAR(state);
    }

    flint_printf("bits\ttiming|split\tfaster?\ttiming|split\tfaster?\n");
    for (slong i=0; i < 28; i++)
    {
        printf("%ld\t%.5f|%ld\t%d\t%.5f|%ld\t%d\n", i+4, mins[i], mins_k[i], mins[i]<t[i], mins_kara[i], mins_k_kara[i], mins_kara[i]<t[i]);
    }


    return 0;
}