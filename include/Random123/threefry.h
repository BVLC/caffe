/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef _threefry_dot_h_
#define _threefry_dot_h_
#include "features/compilerfeatures.h"
#include "array.h"

/** \cond HIDDEN_FROM_DOXYGEN */
/* Significant parts of this file were copied from
   from:
      Skein_FinalRnd/ReferenceImplementation/skein.h
      Skein_FinalRnd/ReferenceImplementation/skein_block.c

   in http://csrc.nist.gov/groups/ST/hash/sha-3/Round3/documents/Skein_FinalRnd.zip

   This file has been modified so that it may no longer perform its originally
   intended function.  If you're looking for a Skein or Threefish source code,
   please consult the original file.

   The original file had the following header:
**************************************************************************
**
** Interface declarations and internal definitions for Skein hashing.
**
** Source code author: Doug Whiting, 2008.
**
** This algorithm and source code is released to the public domain.
**
***************************************************************************

*/

/* See comment at the top of philox.h for the macro pre-process
   strategy. */

/* Rotation constants: */
enum r123_enum_threefry64x4 {
    /* These are the R_256 constants from the Threefish reference sources
       with names changed to R_64x4... */
    R_64x4_0_0=14, R_64x4_0_1=16,
    R_64x4_1_0=52, R_64x4_1_1=57,
    R_64x4_2_0=23, R_64x4_2_1=40,
    R_64x4_3_0= 5, R_64x4_3_1=37,
    R_64x4_4_0=25, R_64x4_4_1=33,
    R_64x4_5_0=46, R_64x4_5_1=12,
    R_64x4_6_0=58, R_64x4_6_1=22,
    R_64x4_7_0=32, R_64x4_7_1=32
};

enum r123_enum_threefry64x2 {
    /*
    // Output from skein_rot_search: (srs64_B64-X1000)
    // Random seed = 1. BlockSize = 128 bits. sampleCnt =  1024. rounds =  8, minHW_or=57
    // Start: Tue Mar  1 10:07:48 2011
    // rMin = 0.136. #0325[*15] [CRC=455A682F. hw_OR=64. cnt=16384. blkSize= 128].format   
    */
    R_64x2_0_0=16,
    R_64x2_1_0=42,
    R_64x2_2_0=12,
    R_64x2_3_0=31,
    R_64x2_4_0=16,
    R_64x2_5_0=32,
    R_64x2_6_0=24,
    R_64x2_7_0=21
    /* 4 rounds: minHW =  4  [  4  4  4  4 ]
    // 5 rounds: minHW =  8  [  8  8  8  8 ]
    // 6 rounds: minHW = 16  [ 16 16 16 16 ]
    // 7 rounds: minHW = 32  [ 32 32 32 32 ]
    // 8 rounds: minHW = 64  [ 64 64 64 64 ]
    // 9 rounds: minHW = 64  [ 64 64 64 64 ]
    //10 rounds: minHW = 64  [ 64 64 64 64 ]
    //11 rounds: minHW = 64  [ 64 64 64 64 ] */
};

enum r123_enum_threefry32x4 {
    /* Output from skein_rot_search: (srs-B128-X5000.out)
    // Random seed = 1. BlockSize = 64 bits. sampleCnt =  1024. rounds =  8, minHW_or=28
    // Start: Mon Aug 24 22:41:36 2009
    // ...
    // rMin = 0.472. #0A4B[*33] [CRC=DD1ECE0F. hw_OR=31. cnt=16384. blkSize= 128].format    */
    R_32x4_0_0=10, R_32x4_0_1=26,
    R_32x4_1_0=11, R_32x4_1_1=21,
    R_32x4_2_0=13, R_32x4_2_1=27,
    R_32x4_3_0=23, R_32x4_3_1= 5,
    R_32x4_4_0= 6, R_32x4_4_1=20,
    R_32x4_5_0=17, R_32x4_5_1=11,
    R_32x4_6_0=25, R_32x4_6_1=10,
    R_32x4_7_0=18, R_32x4_7_1=20

    /* 4 rounds: minHW =  3  [  3  3  3  3 ]
    // 5 rounds: minHW =  7  [  7  7  7  7 ]
    // 6 rounds: minHW = 12  [ 13 12 13 12 ]
    // 7 rounds: minHW = 22  [ 22 23 22 23 ]
    // 8 rounds: minHW = 31  [ 31 31 31 31 ]
    // 9 rounds: minHW = 32  [ 32 32 32 32 ]
    //10 rounds: minHW = 32  [ 32 32 32 32 ]
    //11 rounds: minHW = 32  [ 32 32 32 32 ] */

};

enum r123_enum_threefry32x2 {
    /* Output from skein_rot_search (srs32x2-X5000.out)
    // Random seed = 1. BlockSize = 64 bits. sampleCnt =  1024. rounds =  8, minHW_or=28
    // Start: Tue Jul 12 11:11:33 2011
    // rMin = 0.334. #0206[*07] [CRC=1D9765C0. hw_OR=32. cnt=16384. blkSize=  64].format   */
    R_32x2_0_0=13,
    R_32x2_1_0=15,
    R_32x2_2_0=26,
    R_32x2_3_0= 6,
    R_32x2_4_0=17,
    R_32x2_5_0=29,
    R_32x2_6_0=16,
    R_32x2_7_0=24

    /* 4 rounds: minHW =  4  [  4  4  4  4 ]
    // 5 rounds: minHW =  6  [  6  8  6  8 ]
    // 6 rounds: minHW =  9  [  9 12  9 12 ]
    // 7 rounds: minHW = 16  [ 16 24 16 24 ]
    // 8 rounds: minHW = 32  [ 32 32 32 32 ]
    // 9 rounds: minHW = 32  [ 32 32 32 32 ]
    //10 rounds: minHW = 32  [ 32 32 32 32 ]
    //11 rounds: minHW = 32  [ 32 32 32 32 ] */
    };

enum r123_enum_threefry_wcnt {
    WCNT2=2,
    WCNT4=4
};
R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(uint64_t RotL_64(uint64_t x, unsigned int N));
R123_CUDA_DEVICE R123_STATIC_INLINE uint64_t RotL_64(uint64_t x, unsigned int N)
{
    return (x << (N & 63)) | (x >> ((64-N) & 63));
}
    
R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(uint32_t RotL_32(uint32_t x, unsigned int N));
R123_CUDA_DEVICE R123_STATIC_INLINE uint32_t RotL_32(uint32_t x, unsigned int N)
{
    return (x << (N & 31)) | (x >> ((32-N) & 31));
}

#define SKEIN_MK_64(hi32,lo32)  ((lo32) + (((uint64_t) (hi32)) << 32))
#define SKEIN_KS_PARITY64         SKEIN_MK_64(0x1BD11BDA,0xA9FC1A22)
#define SKEIN_KS_PARITY32         0x1BD11BDA

#ifndef THREEFRY2x32_DEFAULT_ROUNDS
#define THREEFRY2x32_DEFAULT_ROUNDS 20
#endif

#ifndef THREEFRY2x64_DEFAULT_ROUNDS
#define THREEFRY2x64_DEFAULT_ROUNDS 20
#endif

#ifndef THREEFRY4x32_DEFAULT_ROUNDS
#define THREEFRY4x32_DEFAULT_ROUNDS 20
#endif

#ifndef THREEFRY4x64_DEFAULT_ROUNDS
#define THREEFRY4x64_DEFAULT_ROUNDS 20
#endif

#define _threefry2x_tpl(W)                                              \
typedef struct r123array2x##W threefry2x##W##_ctr_t;                          \
typedef struct r123array2x##W threefry2x##W##_key_t;                          \
typedef struct r123array2x##W threefry2x##W##_ukey_t;                          \
R123_CUDA_DEVICE R123_STATIC_INLINE threefry2x##W##_key_t threefry2x##W##keyinit(threefry2x##W##_ukey_t uk) { return uk; } \
R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(threefry2x##W##_ctr_t threefry2x##W##_R(unsigned int Nrounds, threefry2x##W##_ctr_t in, threefry2x##W##_key_t k)); \
R123_CUDA_DEVICE R123_STATIC_INLINE                                          \
threefry2x##W##_ctr_t threefry2x##W##_R(unsigned int Nrounds, threefry2x##W##_ctr_t in, threefry2x##W##_key_t k){ \
    threefry2x##W##_ctr_t X;                                              \
    uint##W##_t ks[2+1];                                          \
    int  i; /* avoid size_t to avoid need for stddef.h */                   \
    R123_ASSERT(Nrounds<=32);                                           \
    ks[2] =  SKEIN_KS_PARITY##W;                                   \
    for (i=0;i < 2; i++)                                        \
        {                                                               \
            ks[i] = k.v[i];                                             \
            X.v[i]  = in.v[i];                                          \
            ks[2] ^= k.v[i];                                    \
        }                                                               \
                                                                        \
    /* Insert initial key before round 0 */                             \
    X.v[0] += ks[0]; X.v[1] += ks[1];                                   \
                                                                        \
    if(Nrounds>0){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_0_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>1){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_1_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>2){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_2_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>3){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_3_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>3){                                                      \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[1]; X.v[1] += ks[2];                               \
        X.v[1] += 1;     /* X.v[2-1] += r  */                   \
    }                                                                   \
    if(Nrounds>4){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_4_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>5){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_5_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>6){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_6_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>7){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_7_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>7){                                                      \
        /* InjectKey(r=2) */                                            \
        X.v[0] += ks[2]; X.v[1] += ks[0];                               \
        X.v[1] += 2;                                                    \
    }                                                                   \
    if(Nrounds>8){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_0_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>9){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_1_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>10){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_2_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>11){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_3_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>11){                                                     \
        /* InjectKey(r=3) */                                            \
        X.v[0] += ks[0]; X.v[1] += ks[1];                               \
        X.v[1] += 3;                                                    \
    }                                                                   \
    if(Nrounds>12){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_4_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>13){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_5_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>14){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_6_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>15){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_7_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>15){                                                     \
        /* InjectKey(r=4) */                                            \
        X.v[0] += ks[1]; X.v[1] += ks[2];                               \
        X.v[1] += 4;                                                    \
    }                                                                   \
    if(Nrounds>16){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_0_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>17){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_1_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>18){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_2_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>19){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_3_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>19){                                                     \
        /* InjectKey(r=5) */                                            \
        X.v[0] += ks[2]; X.v[1] += ks[0];                               \
        X.v[1] += 5;                                                    \
    }                                                                   \
    if(Nrounds>20){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_4_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>21){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_5_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>22){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_6_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>23){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_7_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>23){                                                     \
        /* InjectKey(r=6) */                                            \
        X.v[0] += ks[0]; X.v[1] += ks[1];                               \
        X.v[1] += 6;                                                    \
    }                                                                   \
    if(Nrounds>24){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_0_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>25){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_1_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>26){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_2_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>27){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_3_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>27){                                                     \
        /* InjectKey(r=7) */                                            \
        X.v[0] += ks[1]; X.v[1] += ks[2];                               \
        X.v[1] += 7;                                                    \
    }                                                                   \
    if(Nrounds>28){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_4_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>29){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_5_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>30){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_6_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>31){  X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x2_7_0); X.v[1] ^= X.v[0]; } \
    if(Nrounds>31){                                                     \
        /* InjectKey(r=8) */                                            \
        X.v[0] += ks[2]; X.v[1] += ks[0];                               \
        X.v[1] += 8;                                                    \
    }                                                                   \
    return X;                                                           \
}                                                                       \
 /** @ingroup ThreefryNxW */                                            \
enum r123_enum_threefry2x##W { threefry2x##W##_rounds = THREEFRY2x##W##_DEFAULT_ROUNDS };       \
R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(threefry2x##W##_ctr_t threefry2x##W(threefry2x##W##_ctr_t in, threefry2x##W##_key_t k)); \
R123_CUDA_DEVICE R123_STATIC_INLINE                                     \
threefry2x##W##_ctr_t threefry2x##W(threefry2x##W##_ctr_t in, threefry2x##W##_key_t k){ \
    return threefry2x##W##_R(threefry2x##W##_rounds, in, k);            \
}


#define _threefry4x_tpl(W)                                              \
typedef struct r123array4x##W threefry4x##W##_ctr_t;                        \
typedef struct r123array4x##W threefry4x##W##_key_t;                        \
typedef struct r123array4x##W threefry4x##W##_ukey_t;                        \
R123_CUDA_DEVICE R123_STATIC_INLINE threefry4x##W##_key_t threefry4x##W##keyinit(threefry4x##W##_ukey_t uk) { return uk; } \
R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(threefry4x##W##_ctr_t threefry4x##W##_R(unsigned int Nrounds, threefry4x##W##_ctr_t in, threefry4x##W##_key_t k)); \
R123_CUDA_DEVICE R123_STATIC_INLINE                                          \
threefry4x##W##_ctr_t threefry4x##W##_R(unsigned int Nrounds, threefry4x##W##_ctr_t in, threefry4x##W##_key_t k){ \
    threefry4x##W##_ctr_t X;                                            \
    uint##W##_t ks[4+1];                                            \
    int  i; /* avoid size_t to avoid need for stddef.h */                   \
    R123_ASSERT(Nrounds<=72);                                           \
    ks[4] =  SKEIN_KS_PARITY##W;                                    \
    for (i=0;i < 4; i++)                                            \
        {                                                               \
            ks[i] = k.v[i];                                             \
            X.v[i]  = in.v[i];                                          \
            ks[4] ^= k.v[i];                                        \
        }                                                               \
                                                                        \
    /* Insert initial key before round 0 */                             \
    X.v[0] += ks[0]; X.v[1] += ks[1]; X.v[2] += ks[2]; X.v[3] += ks[3]; \
                                                                        \
    if(Nrounds>0){                                                      \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_0_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_0_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>1){                                                      \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_1_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_1_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>2){                                                      \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_2_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_2_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>3){                                                      \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_3_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_3_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>3){                                                      \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[1]; X.v[1] += ks[2]; X.v[2] += ks[3]; X.v[3] += ks[4]; \
        X.v[4-1] += 1;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>4){                                                      \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_4_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_4_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>5){                                                      \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_5_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_5_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>6){                                                      \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_6_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_6_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>7){                                                      \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_7_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_7_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>7){                                                      \
        /* InjectKey(r=2) */                                            \
        X.v[0] += ks[2]; X.v[1] += ks[3]; X.v[2] += ks[4]; X.v[3] += ks[0]; \
        X.v[4-1] += 2;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>8){                                                      \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_0_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_0_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>9){                                                      \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_1_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_1_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>10){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_2_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_2_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>11){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_3_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_3_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>11){                                                     \
        /* InjectKey(r=3) */                                            \
        X.v[0] += ks[3]; X.v[1] += ks[4]; X.v[2] += ks[0]; X.v[3] += ks[1]; \
        X.v[4-1] += 3;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>12){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_4_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_4_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>13){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_5_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_5_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>14){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_6_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_6_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>15){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_7_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_7_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>15){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[4]; X.v[1] += ks[0]; X.v[2] += ks[1]; X.v[3] += ks[2]; \
        X.v[4-1] += 4;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>16){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_0_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_0_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>17){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_1_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_1_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>18){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_2_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_2_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>19){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_3_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_3_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>19){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[0]; X.v[1] += ks[1]; X.v[2] += ks[2]; X.v[3] += ks[3]; \
        X.v[4-1] += 5;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>20){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_4_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_4_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>21){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_5_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_5_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>22){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_6_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_6_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>23){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_7_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_7_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>23){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[1]; X.v[1] += ks[2]; X.v[2] += ks[3]; X.v[3] += ks[4]; \
        X.v[4-1] += 6;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>24){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_0_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_0_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>25){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_1_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_1_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>26){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_2_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_2_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>27){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_3_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_3_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>27){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[2]; X.v[1] += ks[3]; X.v[2] += ks[4]; X.v[3] += ks[0]; \
        X.v[4-1] += 7;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>28){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_4_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_4_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>29){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_5_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_5_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>30){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_6_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_6_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>31){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_7_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_7_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>31){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[3]; X.v[1] += ks[4]; X.v[2] += ks[0]; X.v[3] += ks[1]; \
        X.v[4-1] += 8;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>32){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_0_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_0_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>33){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_1_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_1_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>34){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_2_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_2_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>35){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_3_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_3_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>35){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[4]; X.v[1] += ks[0]; X.v[2] += ks[1]; X.v[3] += ks[2]; \
        X.v[4-1] += 9;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>36){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_4_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_4_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>37){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_5_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_5_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>38){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_6_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_6_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>39){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_7_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_7_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>39){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[0]; X.v[1] += ks[1]; X.v[2] += ks[2]; X.v[3] += ks[3]; \
        X.v[4-1] += 10;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>40){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_0_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_0_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>41){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_1_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_1_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>42){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_2_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_2_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>43){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_3_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_3_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>43){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[1]; X.v[1] += ks[2]; X.v[2] += ks[3]; X.v[3] += ks[4]; \
        X.v[4-1] += 11;     /* X.v[WCNT4-1] += r  */                \
    }                                                                   \
                                                                        \
    if(Nrounds>44){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_4_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_4_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>45){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_5_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_5_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>46){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_6_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_6_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>47){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_7_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_7_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>47){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[2]; X.v[1] += ks[3]; X.v[2] += ks[4]; X.v[3] += ks[0]; \
        X.v[4-1] += 12;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>48){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_0_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_0_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>49){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_1_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_1_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>50){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_2_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_2_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>51){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_3_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_3_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>51){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[3]; X.v[1] += ks[4]; X.v[2] += ks[0]; X.v[3] += ks[1]; \
        X.v[4-1] += 13;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>52){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_4_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_4_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>53){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_5_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_5_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>54){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_6_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_6_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>55){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_7_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_7_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>55){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[4]; X.v[1] += ks[0]; X.v[2] += ks[1]; X.v[3] += ks[2]; \
        X.v[4-1] += 14;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>56){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_0_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_0_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>57){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_1_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_1_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>58){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_2_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_2_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>59){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_3_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_3_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>59){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[0]; X.v[1] += ks[1]; X.v[2] += ks[2]; X.v[3] += ks[3]; \
        X.v[4-1] += 15;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>60){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_4_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_4_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>61){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_5_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_5_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>62){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_6_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_6_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>63){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_7_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_7_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>63){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[1]; X.v[1] += ks[2]; X.v[2] += ks[3]; X.v[3] += ks[4]; \
        X.v[4-1] += 16;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>64){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_0_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_0_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>65){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_1_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_1_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>66){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_2_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_2_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>67){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_3_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_3_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>67){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[2]; X.v[1] += ks[3]; X.v[2] += ks[4]; X.v[3] += ks[0]; \
        X.v[4-1] += 17;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    if(Nrounds>68){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_4_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_4_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>69){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_5_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_5_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>70){                                                     \
        X.v[0] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_6_0); X.v[1] ^= X.v[0]; \
        X.v[2] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_6_1); X.v[3] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>71){                                                     \
        X.v[0] += X.v[3]; X.v[3] = RotL_##W(X.v[3],R_##W##x4_7_0); X.v[3] ^= X.v[0]; \
        X.v[2] += X.v[1]; X.v[1] = RotL_##W(X.v[1],R_##W##x4_7_1); X.v[1] ^= X.v[2]; \
    }                                                                   \
    if(Nrounds>71){                                                     \
        /* InjectKey(r=1) */                                            \
        X.v[0] += ks[3]; X.v[1] += ks[4]; X.v[2] += ks[0]; X.v[3] += ks[1]; \
        X.v[4-1] += 18;     /* X.v[WCNT4-1] += r  */                 \
    }                                                                   \
                                                                        \
    return X;                                                           \
}                                                                       \
 /** @ingroup ThreefryNxW */                                            \
enum r123_enum_threefry4x##W { threefry4x##W##_rounds = THREEFRY4x##W##_DEFAULT_ROUNDS };       \
R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(threefry4x##W##_ctr_t threefry4x##W(threefry4x##W##_ctr_t in, threefry4x##W##_key_t k)); \
R123_CUDA_DEVICE R123_STATIC_INLINE                                     \
threefry4x##W##_ctr_t threefry4x##W(threefry4x##W##_ctr_t in, threefry4x##W##_key_t k){ \
    return threefry4x##W##_R(threefry4x##W##_rounds, in, k);            \
}
/** \endcond */

_threefry2x_tpl(64)
_threefry2x_tpl(32)
_threefry4x_tpl(64)
_threefry4x_tpl(32)

/* gcc4.5 and 4.6 seem to optimize a macro-ized threefryNxW better
   than a static inline function.  Why?  */
#define threefry2x32(c,k) threefry2x32_R(threefry2x32_rounds, c, k)
#define threefry4x32(c,k) threefry4x32_R(threefry4x32_rounds, c, k)
#define threefry2x64(c,k) threefry2x64_R(threefry2x64_rounds, c, k)
#define threefry4x64(c,k) threefry4x64_R(threefry4x64_rounds, c, k)

#ifdef __cplusplus
/** \cond HIDDEN_FROM_DOXYGEN */
#define _threefryNxWclass_tpl(NxW)                                      \
namespace r123{                                                     \
template<unsigned int R>                                                  \
 struct Threefry##NxW##_R{                                              \
    typedef threefry##NxW##_ctr_t ctr_type;                             \
    typedef threefry##NxW##_key_t key_type;                             \
    typedef threefry##NxW##_key_t ukey_type;                            \
    static const unsigned int rounds=R;                                 \
   inline R123_CUDA_DEVICE R123_FORCE_INLINE(ctr_type operator()(ctr_type ctr, key_type key)){ \
        R123_STATIC_ASSERT(R<=72, "threefry is only unrolled up to 72 rounds\n"); \
        return threefry##NxW##_R(R, ctr, key);                              \
    }                                                                   \
};                                                                      \
 typedef Threefry##NxW##_R<threefry##NxW##_rounds> Threefry##NxW;       \
} // namespace r123

/** \endcond */

_threefryNxWclass_tpl(2x32)
_threefryNxWclass_tpl(4x32)
_threefryNxWclass_tpl(2x64)
_threefryNxWclass_tpl(4x64)

/* The _tpl macros don't quite work to do string-pasting inside comments.
   so we just write out the boilerplate documentation four times... */

/** 
@defgroup ThreefryNxW Threefry Classes and Typedefs

The ThreefryNxW classes export the member functions, typedefs and
operator overloads required by a @ref CBRNG "CBRNG" class.

As described in  
<a href="http://dl.acm.org/citation.cfm?doid=2063405"><i>Parallel Random Numbers:  As Easy as 1, 2, 3</i> </a>, 
the Threefry family is closely related to the Threefish block cipher from
<a href="http://www.skein-hash.info/"> Skein Hash Function</a>.  
Threefry is \b not suitable for cryptographic use.

Threefry uses integer addition, bitwise rotation, xor and permutation of words to randomize its output.

@class r123::Threefry2x32_R 
@ingroup ThreefryNxW

exports the member functions, typedefs and operator overloads required by a @ref CBRNG "CBRNG" class.

The template argument, ROUNDS, is the number of times the Threefry round
function will be applied.

As of September 2011, the authors know of no statistical flaws with
ROUNDS=13 or more for Threefry2x32.

@typedef r123::Threefry2x32
@ingroup ThreefryNxW
  Threefry2x32 is equivalent to Threefry2x32_R<20>.    With 20 rounds,
  Threefry2x32 has a considerable safety margin over the minimum number
  of rounds with no known statistical flaws, but still has excellent
   performance. 

@class r123::Threefry2x64_R 
@ingroup ThreefryNxW

exports the member functions, typedefs and operator overloads required by a @ref CBRNG "CBRNG" class.

The template argument, ROUNDS, is the number of times the Threefry round
function will be applied.

In November 2011, the authors discovered that 13 rounds of
Threefry2x64 sequenced by strided, interleaved key and counter
increments failed a very long (longer than the default BigCrush
length) WeightDistrub test.  At the same time, it was confirmed that
14 rounds passes much longer tests (up to 5x10^12 samples) of a
similar nature.  The authors know of no statistical flaws with
ROUNDS=14 or more for Threefry2x64.

@typedef r123::Threefry2x64
@ingroup ThreefryNxW
  Threefry2x64 is equivalent to Threefry2x64_R<20>.    With 20 rounds,
  Threefry2x64 has a considerable safety margin over the minimum number
  of rounds with no known statistical flaws, but still has excellent
   performance. 



@class r123::Threefry4x32_R 
@ingroup ThreefryNxW

exports the member functions, typedefs and operator overloads required by a @ref CBRNG "CBRNG" class.

The template argument, ROUNDS, is the number of times the Threefry round
function will be applied.

As of September 2011, the authors know of no statistical flaws with
ROUNDS=12 or more for Threefry4x32.

@typedef r123::Threefry4x32
@ingroup ThreefryNxW
  Threefry4x32 is equivalent to Threefry4x32_R<20>.    With 20 rounds,
  Threefry4x32 has a considerable safety margin over the minimum number
  of rounds with no known statistical flaws, but still has excellent
   performance. 



@class r123::Threefry4x64_R 
@ingroup ThreefryNxW

exports the member functions, typedefs and operator overloads required by a @ref CBRNG "CBRNG" class.

The template argument, ROUNDS, is the number of times the Threefry round
function will be applied.

As of September 2011, the authors know of no statistical flaws with
ROUNDS=12 or more for Threefry4x64.

@typedef r123::Threefry4x64
@ingroup ThreefryNxW
  Threefry4x64 is equivalent to Threefry4x64_R<20>.    With 20 rounds,
  Threefry4x64 has a considerable safety margin over the minimum number
  of rounds with no known statistical flaws, but still has excellent
   performance. 
*/

#endif

#endif
