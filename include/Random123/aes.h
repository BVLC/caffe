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
#ifndef __Random123_aes_dot_hpp__
#define __Random123_aes_dot_hpp__

#include "features/compilerfeatures.h"
#include "array.h"

/* Implement a bona fide AES block cipher.  It's minimally
// checked against the test vector in FIPS-197 in ut_aes.cpp. */
#if R123_USE_AES_NI

/** @ingroup AESNI */
typedef struct r123array1xm128i aesni1xm128i_ctr_t;
/** @ingroup AESNI */
typedef struct r123array1xm128i aesni1xm128i_ukey_t;
/** @ingroup AESNI */
typedef struct r123array4x32 aesni4x32_ukey_t;
/** @ingroup AESNI */
enum r123_enum_aesni1xm128i { aesni1xm128i_rounds = 10 };

/** \cond HIDDEN_FROM_DOXYGEN */
R123_STATIC_INLINE __m128i AES_128_ASSIST (__m128i temp1, __m128i temp2) { 
    __m128i temp3; 
    temp2 = _mm_shuffle_epi32 (temp2 ,0xff); 
    temp3 = _mm_slli_si128 (temp1, 0x4);
    temp1 = _mm_xor_si128 (temp1, temp3);
    temp3 = _mm_slli_si128 (temp3, 0x4);
    temp1 = _mm_xor_si128 (temp1, temp3);
    temp3 = _mm_slli_si128 (temp3, 0x4);
    temp1 = _mm_xor_si128 (temp1, temp3);
    temp1 = _mm_xor_si128 (temp1, temp2); 
    return temp1; 
}

R123_STATIC_INLINE void aesni1xm128iexpand(aesni1xm128i_ukey_t uk, __m128i ret[11])
{
    __m128i rkey = uk.v[0].m;
    __m128i tmp2;

    ret[0] = rkey;
    tmp2 = _mm_aeskeygenassist_si128(rkey, 0x1);
    rkey = AES_128_ASSIST(rkey, tmp2);
    ret[1] = rkey;

    tmp2 = _mm_aeskeygenassist_si128(rkey, 0x2);
    rkey = AES_128_ASSIST(rkey, tmp2);
    ret[2] = rkey;

    tmp2 = _mm_aeskeygenassist_si128(rkey, 0x4);
    rkey = AES_128_ASSIST(rkey, tmp2);
    ret[3] = rkey;

    tmp2 = _mm_aeskeygenassist_si128(rkey, 0x8);
    rkey = AES_128_ASSIST(rkey, tmp2);
    ret[4] = rkey;

    tmp2 = _mm_aeskeygenassist_si128(rkey, 0x10);
    rkey = AES_128_ASSIST(rkey, tmp2);
    ret[5] = rkey;

    tmp2 = _mm_aeskeygenassist_si128(rkey, 0x20);
    rkey = AES_128_ASSIST(rkey, tmp2);
    ret[6] = rkey;

    tmp2 = _mm_aeskeygenassist_si128(rkey, 0x40);
    rkey = AES_128_ASSIST(rkey, tmp2);
    ret[7] = rkey;

    tmp2 = _mm_aeskeygenassist_si128(rkey, 0x80);
    rkey = AES_128_ASSIST(rkey, tmp2);
    ret[8] = rkey;

    tmp2 = _mm_aeskeygenassist_si128(rkey, 0x1b);
    rkey = AES_128_ASSIST(rkey, tmp2);
    ret[9] = rkey;

    tmp2 = _mm_aeskeygenassist_si128(rkey, 0x36);
    rkey = AES_128_ASSIST(rkey, tmp2);
    ret[10] = rkey;
}
/** \endcond */
    
#ifdef __cplusplus
/** @ingroup AESNI */
struct aesni1xm128i_key_t{ 
    __m128i k[11]; 
    aesni1xm128i_key_t(){
        aesni1xm128i_ukey_t uk;
        uk.v[0].m = _mm_setzero_si128();
        aesni1xm128iexpand(uk, k);
    }
    aesni1xm128i_key_t(const aesni1xm128i_ukey_t& uk){
        aesni1xm128iexpand(uk, k);
    }
    aesni1xm128i_key_t(const aesni4x32_ukey_t& uk){
        aesni1xm128i_ukey_t uk128;
        uk128.v[0].m = _mm_set_epi32(uk.v[3], uk.v[2], uk.v[1], uk.v[0]);
        aesni1xm128iexpand(uk128, k);
    }
    aesni1xm128i_key_t& operator=(const aesni1xm128i_ukey_t& uk){
        aesni1xm128iexpand(uk, k);
        return *this;
    }
    aesni1xm128i_key_t& operator=(const aesni4x32_ukey_t& uk){
        aesni1xm128i_ukey_t uk128;
        uk128.v[0].m = _mm_set_epi32(uk.v[3], uk.v[2], uk.v[1], uk.v[0]);
        aesni1xm128iexpand(uk128, k);
        return *this;
    }
};
#else
typedef struct { 
    __m128i k[11]; 
}aesni1xm128i_key_t;

/** @ingroup AESNI */
R123_STATIC_INLINE aesni1xm128i_key_t aesni1xm128ikeyinit(aesni1xm128i_ukey_t uk){
    aesni1xm128i_key_t ret;
    aesni1xm128iexpand(uk, ret.k);
    return ret;
}
#endif

/** @ingroup AESNI */
R123_STATIC_INLINE aesni1xm128i_ctr_t aesni1xm128i(aesni1xm128i_ctr_t in, aesni1xm128i_key_t k) {
    __m128i x = _mm_xor_si128(k.k[0], in.v[0].m);
    x = _mm_aesenc_si128(x, k.k[1]);
    x = _mm_aesenc_si128(x, k.k[2]);
    x = _mm_aesenc_si128(x, k.k[3]);
    x = _mm_aesenc_si128(x, k.k[4]);
    x = _mm_aesenc_si128(x, k.k[5]);
    x = _mm_aesenc_si128(x, k.k[6]);
    x = _mm_aesenc_si128(x, k.k[7]);
    x = _mm_aesenc_si128(x, k.k[8]);
    x = _mm_aesenc_si128(x, k.k[9]);
    x = _mm_aesenclast_si128(x, k.k[10]);
    {
      aesni1xm128i_ctr_t ret;
      ret.v[0].m = x;
      return ret;
    }
}

/** @ingroup AESNI */
R123_STATIC_INLINE aesni1xm128i_ctr_t aesni1xm128i_R(unsigned R, aesni1xm128i_ctr_t in, aesni1xm128i_key_t k){
    R123_ASSERT(R==10);
    return aesni1xm128i(in, k);
}


/** @ingroup AESNI */
typedef struct r123array4x32 aesni4x32_ctr_t;
/** @ingroup AESNI */
typedef aesni1xm128i_key_t aesni4x32_key_t;
/** @ingroup AESNI */
enum r123_enum_aesni4x32 { aesni4x32_rounds = 10 };
/** @ingroup AESNI */
R123_STATIC_INLINE aesni4x32_key_t aesni4x32keyinit(aesni4x32_ukey_t uk){
    aesni1xm128i_ukey_t uk128;
    aesni4x32_key_t ret;
    uk128.v[0].m = _mm_set_epi32(uk.v[3], uk.v[2], uk.v[1], uk.v[0]);
    aesni1xm128iexpand(uk128, ret.k);
    return ret;
}

/** @ingroup AESNI */
/** The aesni4x32_R function provides a C API to the @ref AESNI "AESNI" CBRNG, allowing the number of rounds to be specified explicitly **/
R123_STATIC_INLINE aesni4x32_ctr_t aesni4x32_R(unsigned int Nrounds, aesni4x32_ctr_t c, aesni4x32_key_t k){
    aesni1xm128i_ctr_t c128;
    c128.v[0].m = _mm_set_epi32(c.v[3], c.v[2], c.v[1], c.v[0]);
    c128 = aesni1xm128i_R(Nrounds, c128, k);
    _mm_storeu_si128((__m128i*)&c.v[0], c128.v[0].m);
    return c;
}

#define aesni4x32_rounds aesni1xm128i_rounds

/** The aesni4x32 macro provides a C API to the @ref AESNI "AESNI" CBRNG, uses the default number of rounds i.e. \c aesni4x32_rounds **/
/** @ingroup AESNI */
#define aesni4x32(c,k) aesni4x32_R(aesni4x32_rounds, c, k)

#ifdef __cplusplus
namespace r123{
/** 
@defgroup AESNI ARS and AESNI Classes and Typedefs

The ARS4x32, ARS1xm128i, AESNI4x32 and AESNI1xm128i classes export the member functions, typedefs and
operator overloads required by a @ref CBRNG "CBRNG" class.

ARS1xm128i and AESNI1xm128i are based on the AES block cipher and rely on the AES-NI hardware instructions
available on some some new (2011) CPUs.

The ARS1xm128i CBRNG and the use of AES for random number generation are described in 
<a href="http://dl.acm.org/citation.cfm?doid=2063405"><i>Parallel Random Numbers:  As Easy as 1, 2, 3</i> </a>.
Although it uses some cryptographic primitives, ARS1xm128i uses a cryptographically weak key schedule and is \b not suitable for cryptographic use.

@class AESNI1xm128i
@ingroup AESNI
AESNI exports the member functions, typedefs and operator overloads required by a @ref CBRNG class.

AESNI1xm128i uses the crypotgraphic AES round function, including the cryptographic key schedule.

In contrast to the other CBRNGs in the Random123 library, the AESNI1xm128i_R::key_type is opaque
and is \b not identical to the AESNI1xm128i_R::ukey_type.  Creating a key_type, using either the constructor
or assignment operator, is significantly more time-consuming than running the bijection (hundreds
of clock cycles vs. tens of clock cycles).

AESNI1xm128i is only available when the feature-test macro R123_USE_AES_NI is true, which
should occur only when the compiler is configured to generate AES-NI instructions (or
when defaults are overridden by compile-time, compiler-command-line options).

As of September 2011, the authors know of no statistical flaws with AESNI1xm128i.  It
would be an event of major cryptographic note if any such flaws were ever found.
*/
struct AESNI1xm128i{
    typedef aesni1xm128i_ctr_t ctr_type;
    typedef aesni1xm128i_ukey_t ukey_type;
    typedef aesni1xm128i_key_t key_type;
    static const unsigned int rounds=10;
    ctr_type operator()(ctr_type ctr, key_type key) const{
        return aesni1xm128i(ctr, key);
    }
};

/* @class AESNI4x32 */
struct AESNI4x32{
    typedef aesni4x32_ctr_t ctr_type;
    typedef aesni4x32_ukey_t ukey_type;
    typedef aesni4x32_key_t key_type;
    static const unsigned int rounds=10;
    ctr_type operator()(ctr_type ctr, key_type key) const{
        return aesni4x32(ctr, key);
    }
};

/** @ingroup AESNI
    @class AESNI1xm128i_R

AESNI1xm128i_R is provided for completeness, but is only instantiable with ROUNDS=10, in
which case it is identical to AESNI1xm128i */
template <unsigned ROUNDS=10> 
struct AESNI1xm128i_R : public AESNI1xm128i{
    R123_STATIC_ASSERT(ROUNDS==10, "AESNI1xm128i_R<R> is only valid with R=10");
};

/** @class AESNI4x32_R **/
template <unsigned ROUNDS=10> 
struct AESNI4x32_R : public AESNI4x32{
    R123_STATIC_ASSERT(ROUNDS==10, "AESNI4x32_R<R> is only valid with R=10");
};
} // namespace r123
#endif /* __cplusplus */

#endif /* R123_USE_AES_NI */

#if R123_USE_AES_OPENSSL
#include <openssl/aes.h>
typedef struct r123array16x8 aesopenssl16x8_ctr_t;
typedef struct r123array16x8 aesopenssl16x8_ukey_t;
#ifdef __cplusplus
struct aesopenssl16x8_key_t{
    AES_KEY k;
    aesopenssl16x8_key_t(){
        aesopenssl16x8_ukey_t ukey={{}};
        AES_set_encrypt_key((const unsigned char *)&ukey.v[0], 128, &k);
    }
    aesopenssl16x8_key_t(const aesopenssl16x8_ukey_t& ukey){
        AES_set_encrypt_key((const unsigned char *)&ukey.v[0], 128, &k);
    }
    aesopenssl16x8_key_t& operator=(const aesopenssl16x8_ukey_t& ukey){
        AES_set_encrypt_key((const unsigned char *)&ukey.v[0], 128, &k);
        return *this;
    }
};
#else
typedef struct aesopenssl16x8_key_t{
    AES_KEY k;
}aesopenssl16x8_key_t;
R123_STATIC_INLINE struct aesopenssl16x8_key_t aesopenssl16x8keyinit(aesopenssl16x8_ukey_t uk){
    aesopenssl16x8_key_t ret;
    AES_set_encrypt_key((const unsigned char *)&uk.v[0], 128, &ret.k);
    return ret;
}
#endif

R123_STATIC_INLINE R123_FORCE_INLINE(aesopenssl16x8_ctr_t aesopenssl16x8_R(aesopenssl16x8_ctr_t ctr, aesopenssl16x8_key_t key));
R123_STATIC_INLINE
aesopenssl16x8_ctr_t aesopenssl16x8_R(aesopenssl16x8_ctr_t ctr, aesopenssl16x8_key_t key){
    aesopenssl16x8_ctr_t ret;
    AES_encrypt((const unsigned char*)&ctr.v[0], (unsigned char *)&ret.v[0], &key.k);
    return ret;
}

#define aesopenssl16x8_rounds aesni4x32_rounds
#define aesopenssl16x8(c,k) aesopenssl16x8_R(aesopenssl16x8_rounds)

#ifdef __cplusplus
namespace r123{
struct AESOpenSSL16x8{
    typedef aesopenssl16x8_ctr_t ctr_type;
    typedef aesopenssl16x8_key_t key_type;
    typedef aesopenssl16x8_ukey_t ukey_type;
    static const unsigned int rounds=10;
    ctr_type operator()(const ctr_type& in, const key_type& k){
        ctr_type out;
        AES_encrypt((const unsigned char *)&in[0], (unsigned char *)&out[0], &k.k);
        return out;
    }
};
} // namespace r123
#endif /* __cplusplus */
#endif /* R123_USE_AES_OPENSSL */

#endif
