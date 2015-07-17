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
#ifndef _philox_dot_h_
#define _philox_dot_h_

/** \cond HIDDEN_FROM_DOXYGEN */

#include "features/compilerfeatures.h"
#include "array.h"


/*
// Macros _Foo_tpl are code generation 'templates'  They define
// inline functions with names obtained by mangling Foo and the
// macro arguments.  E.g.,
//   _mulhilo_tpl(32, uint32_t, uint64_t)
// expands to a definition of:
//   mulhilo32(uint32_t, uint32_t, uint32_t *, uint32_t *)
// We then 'instantiate the template' to define
// several different functions, e.g.,
//   mulhilo32
//   mulhilo64
// These functions will be visible to user code, and may
// also be used later in subsequent templates and definitions.

// A template for mulhilo using a temporary of twice the word-width.
// Gcc figures out that this can be reduced to a single 'mul' instruction,
// despite the apparent use of double-wide variables, shifts, etc.  It's
// obviously not guaranteed that all compilers will be that smart, so
// other implementations might be preferable, e.g., using an intrinsic
// or an asm block.  On the other hand, for 32-bit multiplies,
// this *is* perfectly standard C99 - any C99 compiler should 
// understand it and produce correct code.  For 64-bit multiplies,
// it's only usable if the compiler recognizes that it can do
// arithmetic on a 128-bit type.  That happens to be true for gcc on
// x86-64, and powerpc64 but not much else.
*/
#define _mulhilo_dword_tpl(W, Word, Dword)                              \
R123_CUDA_DEVICE R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, Word* hip){ \
    Dword product = ((Dword)a)*((Dword)b);                              \
    *hip = product>>W;                                                  \
    return (Word)product;                                               \
}

/*
// A template for mulhilo using gnu-style asm syntax.
// INSN can be "mulw", "mull" or "mulq".  
// FIXME - porting to other architectures, we'll need still-more conditional
// branching here.  Note that intrinsics are usually preferable.
*/
#ifdef __powerpc__
#define _mulhilo_asm_tpl(W, Word, INSN)                         \
R123_STATIC_INLINE Word mulhilo##W(Word ax, Word b, Word *hip){ \
    Word dx = 0;                                                \
    __asm__("\n\t"                                              \
        INSN " %0,%1,%2\n\t"                                    \
        : "=r"(dx)                                              \
        : "r"(b), "r"(ax)                                       \
        );                                                      \
    *hip = dx;                                                  \
    return ax*b;                                                \
}
#else
#define _mulhilo_asm_tpl(W, Word, INSN)                         \
R123_STATIC_INLINE Word mulhilo##W(Word ax, Word b, Word *hip){      \
    Word dx;                                                    \
    __asm__("\n\t"                                              \
        INSN " %2\n\t"                                          \
        : "=a"(ax), "=d"(dx)                                    \
        : "r"(b), "0"(ax)                                       \
        );                                                      \
    *hip = dx;                                                  \
    return ax;                                                  \
}
#endif /* __powerpc__ */

/*
// A template for mulhilo using MSVC-style intrinsics
// For example,_umul128 is an msvc intrinsic, c.f.
// http://msdn.microsoft.com/en-us/library/3dayytw9.aspx
*/
#define _mulhilo_msvc_intrin_tpl(W, Word, INTRIN)               \
R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, Word* hip){       \
    return INTRIN(a, b, hip);                                   \
}

/* N.B.  This really should be called _mulhilo_mulhi_intrin.  It just
   happens that CUDA was the first time we used the idiom. */
#define _mulhilo_cuda_intrin_tpl(W, Word, INTRIN)                       \
R123_CUDA_DEVICE R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, Word* hip){ \
    *hip = INTRIN(a, b);                                                \
    return a*b;                                                         \
}

/*
// A template for mulhilo using only word-size operations and
// C99 operators (no adc, no mulhi).  It
// requires four multiplies and a dozen or so shifts, adds
// and tests.  It's not clear what this is good for, other than
// completeness.  On 32-bit platforms, it could be used to
// implement philoxNx64, but on such platforms both the philoxNx32
// and the threefryNx64 cbrngs are going to have much better
// performance.  It is enabled below by R123_USE_MULHILO64_C99,
// but that is currently (Sep 2011) not set by any of the
// features/XXfeatures.h headers.  It can, of course, be
// set with a compile-time -D option.
*/
#define _mulhilo_c99_tpl(W, Word) \
R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, Word *hip){ \
    const unsigned WHALF = W/2;                                    \
    const Word LOMASK = ((((Word)1)<<WHALF)-1);                    \
    Word lo = a*b;               /* full low multiply */           \
    Word ahi = a>>WHALF;                                           \
    Word alo = a& LOMASK;                                          \
    Word bhi = b>>WHALF;                                           \
    Word blo = b& LOMASK;                                          \
                                                                   \
    Word ahbl = ahi*blo;                                           \
    Word albh = alo*bhi;                                           \
                                                                   \
    Word ahbl_albh = ((ahbl&LOMASK) + (albh&LOMASK));                   \
    Word hi = ahi*bhi + (ahbl>>WHALF) +  (albh>>WHALF);                 \
    hi += ahbl_albh >> WHALF; /* carry from the sum of lo(ahbl) + lo(albh) ) */ \
    /* carry from the sum with alo*blo */                               \
    hi += ((lo >> WHALF) < (ahbl_albh&LOMASK));                         \
    *hip = hi;                                                          \
    return lo;                                                          \
}

/*
// A template for mulhilo on a platform that can't do it
// We could put a C version here, but is it better to run *VERY*
// slowly or to just stop and force the user to find another CBRNG?
*/
#define _mulhilo_fail_tpl(W, Word)                                      \
R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, Word *hip){               \
    R123_STATIC_ASSERT(0, "mulhilo" #W " is not implemented on this machine\n"); \
}

/*
// N.B.  There's an MSVC intrinsic called _emul,
// which *might* compile into better code than
// _mulhilo_dword_tpl 
*/
#if R123_USE_MULHILO32_ASM
#ifdef __powerpc__
_mulhilo_asm_tpl(32, uint32_t, "mulhwu")
#else
_mulhilo_asm_tpl(32, uint32_t, "mull")
#endif /* __powerpc__ */
#else
_mulhilo_dword_tpl(32, uint32_t, uint64_t)
#endif

#if R123_USE_PHILOX_64BIT
#if R123_USE_MULHILO64_ASM
#ifdef __powerpc64__
_mulhilo_asm_tpl(64, uint64_t, "mulhdu")
#else
_mulhilo_asm_tpl(64, uint64_t, "mulq")
#endif /* __powerpc64__ */
#elif R123_USE_MULHILO64_MSVC_INTRIN
_mulhilo_msvc_intrin_tpl(64, uint64_t, _umul128)
#elif R123_USE_MULHILO64_CUDA_INTRIN
_mulhilo_cuda_intrin_tpl(64, uint64_t, __umul64hi)
#elif R123_USE_MULHILO64_OPENCL_INTRIN
_mulhilo_cuda_intrin_tpl(64, uint64_t, mul_hi)
#elif R123_USE_MULHILO64_MULHI_INTRIN
_mulhilo_cuda_intrin_tpl(64, uint64_t, R123_MULHILO64_MULHI_INTRIN)
#elif R123_USE_GNU_UINT128
_mulhilo_dword_tpl(64, uint64_t, __uint128_t)
#elif R123_USE_MULHILO64_C99
_mulhilo_c99_tpl(64, uint64_t)
#else
_mulhilo_fail_tpl(64, uint64_t)
#endif
#endif

/*
// The multipliers and Weyl constants are "hard coded".
// To change them, you can #define them with different
// values before #include-ing this file. 
// This isn't terribly elegant, but it works for C as
// well as C++.  A nice C++-only solution would be to
// use template parameters in the style of <random>
*/
#ifndef PHILOX_M2x64_0
#define PHILOX_M2x64_0 R123_64BIT(0xD2B74407B1CE6E93)
#endif

#ifndef PHILOX_M4x64_0
#define PHILOX_M4x64_0 R123_64BIT(0xD2E7470EE14C6C93)
#endif

#ifndef PHILOX_M4x64_1
#define PHILOX_M4x64_1 R123_64BIT(0xCA5A826395121157)
#endif

#ifndef PHILOX_M2x32_0
#define PHILOX_M2x32_0 ((uint32_t)0xd256d193)
#endif

#ifndef PHILOX_M4x32_0
#define PHILOX_M4x32_0 ((uint32_t)0xD2511F53)
#endif
#ifndef PHILOX_M4x32_1
#define PHILOX_M4x32_1 ((uint32_t)0xCD9E8D57)
#endif

#ifndef PHILOX_W64_0
#define PHILOX_W64_0 R123_64BIT(0x9E3779B97F4A7C15)  /* golden ratio */
#endif
#ifndef PHILOX_W64_1
#define PHILOX_W64_1 R123_64BIT(0xBB67AE8584CAA73B)  /* sqrt(3)-1 */
#endif

#ifndef PHILOX_W32_0
#define PHILOX_W32_0 ((uint32_t)0x9E3779B9)
#endif
#ifndef PHILOX_W32_1
#define PHILOX_W32_1 ((uint32_t)0xBB67AE85)
#endif

#ifndef PHILOX2x32_DEFAULT_ROUNDS
#define PHILOX2x32_DEFAULT_ROUNDS 10
#endif

#ifndef PHILOX2x64_DEFAULT_ROUNDS
#define PHILOX2x64_DEFAULT_ROUNDS 10
#endif

#ifndef PHILOX4x32_DEFAULT_ROUNDS
#define PHILOX4x32_DEFAULT_ROUNDS 10
#endif

#ifndef PHILOX4x64_DEFAULT_ROUNDS
#define PHILOX4x64_DEFAULT_ROUNDS 10
#endif

/* The ignored fourth argument allows us to instantiate the
   same macro regardless of N. */
#define _philox2xWround_tpl(W, T)                                       \
R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(struct r123array2x##W _philox2x##W##round(struct r123array2x##W ctr, struct r123array1x##W key)); \
R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array2x##W _philox2x##W##round(struct r123array2x##W ctr, struct r123array1x##W key){ \
    T hi;                                                               \
    T lo = mulhilo##W(PHILOX_M2x##W##_0, ctr.v[0], &hi);                \
    struct r123array2x##W out = {{hi^key.v[0]^ctr.v[1], lo}};               \
    return out;                                                         \
}
#define _philox2xWbumpkey_tpl(W)                                        \
R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array1x##W _philox2x##W##bumpkey( struct r123array1x##W key) { \
    key.v[0] += PHILOX_W##W##_0;                                        \
    return key;                                                         \
}

#define _philox4xWround_tpl(W, T)                                       \
R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(struct r123array4x##W _philox4x##W##round(struct r123array4x##W ctr, struct r123array2x##W key)); \
R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array4x##W _philox4x##W##round(struct r123array4x##W ctr, struct r123array2x##W key){ \
    T hi0;                                                              \
    T hi1;                                                              \
    T lo0 = mulhilo##W(PHILOX_M4x##W##_0, ctr.v[0], &hi0);              \
    T lo1 = mulhilo##W(PHILOX_M4x##W##_1, ctr.v[2], &hi1);              \
    struct r123array4x##W out = {{hi1^ctr.v[1]^key.v[0], lo1,               \
                              hi0^ctr.v[3]^key.v[1], lo0}};             \
    return out;                                                         \
}

#define _philox4xWbumpkey_tpl(W)                                        \
R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array2x##W _philox4x##W##bumpkey( struct r123array2x##W key) { \
    key.v[0] += PHILOX_W##W##_0;                                        \
    key.v[1] += PHILOX_W##W##_1;                                        \
    return key;                                                         \
}

#define _philoxNxW_tpl(N, Nhalf, W, T)                         \
/** @ingroup PhiloxNxW */                                       \
enum r123_enum_philox##N##x##W { philox##N##x##W##_rounds = PHILOX##N##x##W##_DEFAULT_ROUNDS }; \
typedef struct r123array##N##x##W philox##N##x##W##_ctr_t;                  \
typedef struct r123array##Nhalf##x##W philox##N##x##W##_key_t;              \
typedef struct r123array##Nhalf##x##W philox##N##x##W##_ukey_t;              \
R123_CUDA_DEVICE R123_STATIC_INLINE philox##N##x##W##_key_t philox##N##x##W##keyinit(philox##N##x##W##_ukey_t uk) { return uk; } \
R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(philox##N##x##W##_ctr_t philox##N##x##W##_R(unsigned int R, philox##N##x##W##_ctr_t ctr, philox##N##x##W##_key_t key)); \
R123_CUDA_DEVICE R123_STATIC_INLINE philox##N##x##W##_ctr_t philox##N##x##W##_R(unsigned int R, philox##N##x##W##_ctr_t ctr, philox##N##x##W##_key_t key) { \
    R123_ASSERT(R<=16);                                                 \
    if(R>0){                                       ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>1){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>2){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>3){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>4){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>5){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>6){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>7){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>8){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>9){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>10){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>11){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>12){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>13){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>14){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    if(R>15){ key = _philox##N##x##W##bumpkey(key); ctr = _philox##N##x##W##round(ctr, key); } \
    return ctr;                                                         \
}
         
_philox2xWbumpkey_tpl(32)
_philox4xWbumpkey_tpl(32)
_philox2xWround_tpl(32, uint32_t) /* philo2x32round */
_philox4xWround_tpl(32, uint32_t)            /* philo4x32round */
/** \endcond */
_philoxNxW_tpl(2, 1, 32, uint32_t)    /* philox2x32bijection */
_philoxNxW_tpl(4, 2, 32, uint32_t)    /* philox4x32bijection */
#if R123_USE_PHILOX_64BIT
/** \cond HIDDEN_FROM_DOXYGEN */
_philox2xWbumpkey_tpl(64)
_philox4xWbumpkey_tpl(64)
_philox2xWround_tpl(64, uint64_t) /* philo2x64round */
_philox4xWround_tpl(64, uint64_t) /* philo4x64round */
/** \endcond */
_philoxNxW_tpl(2, 1, 64, uint64_t)    /* philox2x64bijection */
_philoxNxW_tpl(4, 2, 64, uint64_t)    /* philox4x64bijection */
#endif /* R123_USE_PHILOX_64BIT */

#define philox2x32(c,k) philox2x32_R(philox2x32_rounds, c, k)
#define philox4x32(c,k) philox4x32_R(philox4x32_rounds, c, k)
#if R123_USE_PHILOX_64BIT
#define philox2x64(c,k) philox2x64_R(philox2x64_rounds, c, k)
#define philox4x64(c,k) philox4x64_R(philox4x64_rounds, c, k)
#endif /* R123_USE_PHILOX_64BIT */

#ifdef __cplusplus
#include <stdexcept>

/** \cond HIDDEN_FROM_DOXYGEN */

#define _PhiloxNxW_base_tpl(CType, KType, N, W)                         \
namespace r123{                                                          \
template<unsigned int ROUNDS>                                             \
struct Philox##N##x##W##_R{                                             \
    typedef CType ctr_type;                                         \
    typedef KType key_type;                                             \
    typedef KType ukey_type;                                         \
    static const unsigned int rounds=ROUNDS;                                 \
    inline R123_CUDA_DEVICE R123_FORCE_INLINE(ctr_type operator()(ctr_type ctr, key_type key) const){ \
        R123_STATIC_ASSERT(ROUNDS<=16, "philox is only unrolled up to 16 rounds\n"); \
        return philox##N##x##W##_R(ROUNDS, ctr, key);                       \
    }                                                                   \
};                                                                      \
typedef Philox##N##x##W##_R<philox##N##x##W##_rounds> Philox##N##x##W; \
 } // namespace r123
/** \endcond */

_PhiloxNxW_base_tpl(r123array2x32, r123array1x32, 2, 32) // Philox2x32_R<R>
_PhiloxNxW_base_tpl(r123array4x32, r123array2x32, 4, 32) // Philox4x32_R<R>
#if R123_USE_PHILOX_64BIT
_PhiloxNxW_base_tpl(r123array2x64, r123array1x64, 2, 64) // Philox2x64_R<R>
_PhiloxNxW_base_tpl(r123array4x64, r123array2x64, 4, 64) // Philox4x64_R<R>
#endif

/* The _tpl macros don't quite work to do string-pasting inside comments.
   so we just write out the boilerplate documentation four times... */

/** 
@defgroup PhiloxNxW Philox Classes and Typedefs

The PhiloxNxW classes export the member functions, typedefs and
operator overloads required by a @ref CBRNG "CBRNG" class.

As described in  
<a href="http://dl.acm.org/citation.cfm?doid=2063405"><i>Parallel Random Numbers:  As Easy as 1, 2, 3</i> </a>.
The Philox family of counter-based RNGs use integer multiplication, xor and permutation of W-bit words
to scramble its N-word input key.  Philox is a mnemonic for Product HI LO Xor).


@class r123::Philox2x32_R 
@ingroup PhiloxNxW

exports the member functions, typedefs and operator overloads required by a @ref CBRNG "CBRNG" class.

The template argument, ROUNDS, is the number of times the Philox round
function will be applied.

As of November 2011, the authors know of no statistical flaws with
ROUNDS=6 or more for Philox2x32.

@typedef r123::Philox2x32
@ingroup PhiloxNxW
  Philox2x32 is equivalent to Philox2x32_R<10>.    With 10 rounds,
  Philox2x32 has a considerable safety margin over the minimum number
  of rounds with no known statistical flaws, but still has excellent
   performance. 



@class r123::Philox2x64_R 
@ingroup PhiloxNxW

exports the member functions, typedefs and operator overloads required by a @ref CBRNG "CBRNG" class.

The template argument, ROUNDS, is the number of times the Philox round
function will be applied.

As of September 2011, the authors know of no statistical flaws with
ROUNDS=6 or more for Philox2x64.

@typedef r123::Philox2x64
@ingroup PhiloxNxW
  Philox2x64 is equivalent to Philox2x64_R<10>.    With 10 rounds,
  Philox2x64 has a considerable safety margin over the minimum number
  of rounds with no known statistical flaws, but still has excellent
   performance. 



@class r123::Philox4x32_R 
@ingroup PhiloxNxW

exports the member functions, typedefs and operator overloads required by a @ref CBRNG "CBRNG" class.

The template argument, ROUNDS, is the number of times the Philox round
function will be applied.

In November 2011, the authors recorded some suspicious p-values (approximately 1.e-7) from
some very long (longer than the default BigCrush length) SimpPoker tests.  Despite
the fact that even longer tests reverted to "passing" p-values, a cloud remains over
Philox4x32 with 7 rounds.  The authors know of no statistical flaws with
ROUNDS=8 or more for Philox4x32.

@typedef r123::Philox4x32
@ingroup PhiloxNxW
  Philox4x32 is equivalent to Philox4x32_R<10>.    With 10 rounds,
  Philox4x32 has a considerable safety margin over the minimum number
  of rounds with no known statistical flaws, but still has excellent
   performance. 



@class r123::Philox4x64_R 
@ingroup PhiloxNxW

exports the member functions, typedefs and operator overloads required by a @ref CBRNG "CBRNG" class.

The template argument, ROUNDS, is the number of times the Philox round
function will be applied.

As of September 2011, the authors know of no statistical flaws with
ROUNDS=7 or more for Philox4x64.

@typedef r123::Philox4x64
@ingroup PhiloxNxW
  Philox4x64 is equivalent to Philox4x64_R<10>.    With 10 rounds,
  Philox4x64 has a considerable safety margin over the minimum number
  of rounds with no known statistical flaws, but still has excellent
   performance. 
*/

#endif /* __cplusplus */

#endif /* _philox_dot_h_ */
