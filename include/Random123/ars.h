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
#ifndef __Random123_ars_dot_hpp__
#define __Random123_ars_dot_hpp__

#include "features/compilerfeatures.h"
#include "array.h"

#if R123_USE_AES_NI

#ifndef ARS1xm128i_DEFAULT_ROUNDS
#define ARS1xm128i_DEFAULT_ROUNDS 7
#endif

/** @ingroup AESNI */
enum r123_enum_ars1xm128i {ars1xm128i_rounds = ARS1xm128i_DEFAULT_ROUNDS};

/* ARS1xm128i with Weyl keys.  Fast, and Crush-resistant, but NOT CRYPTO. */
/** @ingroup AESNI */
typedef struct r123array1xm128i ars1xm128i_ctr_t;
/** @ingroup AESNI */
typedef struct r123array1xm128i ars1xm128i_key_t;
/** @ingroup AESNI */
typedef struct r123array1xm128i ars1xm128i_ukey_t;
/** @ingroup AESNI */
R123_STATIC_INLINE ars1xm128i_key_t ars1xm128ikeyinit(ars1xm128i_ukey_t uk) { return uk; }
/** @ingroup AESNI */
R123_STATIC_INLINE ars1xm128i_ctr_t ars1xm128i_R(unsigned int Nrounds, ars1xm128i_ctr_t in, ars1xm128i_key_t k){
    __m128i kweyl = _mm_set_epi64x(R123_64BIT(0xBB67AE8584CAA73B), /* sqrt(3) - 1.0 */
                                   R123_64BIT(0x9E3779B97F4A7C15)); /* golden ratio */
    /* N.B.  the aesenc instructions do the xor *after*
    // so if we want to follow the AES pattern, we
    // have to do the initial xor explicitly */
    __m128i kk = k.v[0].m;
    __m128i v = _mm_xor_si128(in.v[0].m, kk);
    ars1xm128i_ctr_t ret;
    R123_ASSERT(Nrounds<=10);
    if( Nrounds>1 ){
        kk = _mm_add_epi64(kk, kweyl);
        v = _mm_aesenc_si128(v, kk);
    }
    if( Nrounds>2 ){
        kk = _mm_add_epi64(kk, kweyl);
        v = _mm_aesenc_si128(v, kk);
    }
    if( Nrounds>3 ){
        kk = _mm_add_epi64(kk, kweyl);
        v = _mm_aesenc_si128(v, kk);
    }
    if( Nrounds>4 ){
        kk = _mm_add_epi64(kk, kweyl);
        v = _mm_aesenc_si128(v, kk);
    }
    if( Nrounds>5 ){
        kk = _mm_add_epi64(kk, kweyl);
        v = _mm_aesenc_si128(v, kk);
    }
    if( Nrounds>6 ){
        kk = _mm_add_epi64(kk, kweyl);
        v = _mm_aesenc_si128(v, kk);
    }
    if( Nrounds>7 ){
        kk = _mm_add_epi64(kk, kweyl);
        v = _mm_aesenc_si128(v, kk);
    }
    if( Nrounds>8 ){
        kk = _mm_add_epi64(kk, kweyl);
        v = _mm_aesenc_si128(v, kk);
    }
    if( Nrounds>9 ){
        kk = _mm_add_epi64(kk, kweyl);
        v = _mm_aesenc_si128(v, kk);
    }
    kk = _mm_add_epi64(kk, kweyl);
    v = _mm_aesenclast_si128(v, kk);
    ret.v[0].m = v;
    return ret;
}

/** @def ars1xm128i
@ingroup AESNI
The ars1mx128i macro provides a C API interface to the @ref AESNI "ARS" CBRNG with the default number of rounds i.e. \c ars1xm128i_rounds **/
#define ars1xm128i(c,k) ars1xm128i_R(ars1xm128i_rounds, c, k)

/** @ingroup AESNI */
typedef struct r123array4x32 ars4x32_ctr_t;
/** @ingroup AESNI */
typedef struct r123array4x32 ars4x32_key_t;
/** @ingroup AESNI */
typedef struct r123array4x32 ars4x32_ukey_t;
/** @ingroup AESNI */
enum r123_enum_ars4x32 {ars4x32_rounds = ARS1xm128i_DEFAULT_ROUNDS};
/** @ingroup AESNI */
R123_STATIC_INLINE ars4x32_key_t ars4x32keyinit(ars4x32_ukey_t uk) { return uk; }
/** @ingroup AESNI */
R123_STATIC_INLINE ars4x32_ctr_t ars4x32_R(unsigned int Nrounds, ars4x32_ctr_t c, ars4x32_key_t k){
    ars1xm128i_ctr_t c128;
    ars1xm128i_key_t k128;
    c128.v[0].m = _mm_set_epi32(c.v[3], c.v[2], c.v[1], c.v[0]);
    k128.v[0].m = _mm_set_epi32(k.v[3], k.v[2], k.v[1], k.v[0]);
    c128 = ars1xm128i_R(Nrounds, c128, k128);
    _mm_storeu_si128((__m128i*)&c.v[0], c128.v[0].m);
    return c;
}

/** @def ars4x32
@ingroup AESNI
The ars4x32 macro provides a C API interface to the @ref AESNI "ARS" CBRNG with the default number of rounds i.e. \c ars4x32_rounds **/
#define ars4x32(c,k) ars4x32_R(ars4x32_rounds, c, k)

#ifdef __cplusplus
namespace r123{
/** 
@ingroup AESNI

ARS1xm128i_R exports the member functions, typedefs and operator overloads required by a @ref CBRNG class.

ARS1xm128i uses the crypotgraphic AES round function, but a @b non-cryptographc key schedule
to save time and space.

ARS1xm128i is only available when the feature-test macro R123_USE_AES_NI is true, which
should occur only when the compiler is configured to generate AES-NI instructions (or
when defaults are overridden by compile-time, compiler-command-line options).

The template argument, ROUNDS, is the number of times the ARS round
functions will be applied.

As of September 2011, the authors know of no statistical flaws with
ROUNDS=5 or more.

@class ARS1xm128i_R

*/
template<unsigned int ROUNDS>
struct ARS1xm128i_R{
    typedef ars1xm128i_ctr_t ctr_type;
    typedef ars1xm128i_key_t key_type;
    typedef ars1xm128i_key_t ukey_type;
    static const unsigned int rounds=ROUNDS;
    R123_FORCE_INLINE(ctr_type operator()(ctr_type ctr, key_type key) const){
        return ars1xm128i_R(ROUNDS, ctr, key);
    }
};

/** @class ARS4x32_R
    @ingroup AESNI
*/

template<unsigned int ROUNDS>
struct ARS4x32_R{
    typedef ars4x32_ctr_t ctr_type;
    typedef ars4x32_key_t key_type;
    typedef ars4x32_key_t ukey_type;
    static const unsigned int rounds=ROUNDS;
    R123_FORCE_INLINE(ctr_type operator()(ctr_type ctr, key_type key) const){
        return ars4x32_R(ROUNDS, ctr, key);
    }
};
/**
@ingroup AESNI

@class ARS1xm128i_R
  ARS1xm128i is equivalent to ARS1xm128i_R<7>.    With 7 rounds,
  the ARS1xm128i CBRNG  has a considerable safety margin over the minimum number
  of rounds with no known statistical flaws, but still has excellent
  performance. */
typedef ARS1xm128i_R<ars1xm128i_rounds> ARS1xm128i;
typedef ARS4x32_R<ars4x32_rounds> ARS4x32;
} // namespace r123

#endif /* __cplusplus */

#endif /* R123_USE_AES_NI */

#endif
