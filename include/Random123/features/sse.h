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
#ifndef _Random123_sse_dot_h__
#define _Random123_sse_dot_h__

#if R123_USE_SSE

#if R123_USE_X86INTRIN_H
#include <x86intrin.h>
#endif
#if R123_USE_IA32INTRIN_H
#include <ia32intrin.h>
#endif
#if R123_USE_XMMINTRIN_H
#include <xmmintrin.h>
#endif
#if R123_USE_EMMINTRIN_H
#include <emmintrin.h>
#endif
#if R123_USE_SMMINTRIN_H
#include <smmintrin.h>
#endif
#if R123_USE_WMMINTRIN_H
#include <wmmintrin.h>
#endif
#if R123_USE_INTRIN_H
#include <intrin.h>
#endif
#ifdef __cplusplus
#include <iostream>
#include <limits>
#include <stdexcept>
#endif

#if R123_USE_ASM_GNU

/* bit25 of CX tells us whether AES is enabled. */
R123_STATIC_INLINE int haveAESNI(){
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__ ("cpuid": "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx) :
                      "a" (1));
    return (ecx>>25) & 1;
}
#elif R123_USE_CPUID_MSVC
R123_STATIC_INLINE int haveAESNI(){
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    return (CPUInfo[2]>>25)&1;
}
#else /* R123_USE_CPUID_??? */
#warning "No R123_USE_CPUID_XXX method chosen.  haveAESNI will always return false"
R123_STATIC_INLINE int haveAESNI(){
    return 0;
}
#endif /* R123_USE_ASM_GNU || R123_USE_CPUID_MSVC */

// There is a lot of annoying and inexplicable variation in the
// SSE intrinsics available in different compilation environments.
// The details seem to depend on the compiler, the version and
// the target architecture.  Rather than insisting on
// R123_USE_feature tests for each of these in each of the
// compilerfeatures.h files we just keep the complexity localized
// to here...
#if (defined(__ICC) && __ICC<1210) || (defined(_MSC_VER) && !defined(_WIN64))
/* Is there an intrinsic to assemble an __m128i from two 64-bit words? 
   If not, use the 4x32-bit intrisic instead.  N.B.  It looks like Intel
   added _mm_set_epi64x to icc version 12.1 in Jan 2012.
*/
R123_STATIC_INLINE __m128i _mm_set_epi64x(uint64_t v1, uint64_t v0){
    union{
        uint64_t u64;
        uint32_t u32[2];
    } u1, u0;
    u1.u64 = v1;
    u0.u64 = v0;
    return _mm_set_epi32(u1.u32[1], u1.u32[0], u0.u32[1], u0.u32[0]);
}
#endif
/* _mm_extract_lo64 abstracts the task of extracting the low 64-bit
   word from an __m128i.  The _mm_cvtsi128_si64 intrinsic does the job
   on 64-bit platforms.  Unfortunately, both MSVC and Open64 fail
   assertions in ut_M128.cpp and ut_carray.cpp when we use the
   _mm_cvtsi128_si64 intrinsic.  (See
   https://bugs.open64.net/show_bug.cgi?id=873 for the Open64 bug).
   On 32-bit platforms, there's no MOVQ, so there's no intrinsic.
   Finally, even if the intrinsic exists, it may be spelled with or
   without the 'x'.
*/
#if !defined(__x86_64__) || defined(_MSC_VER) || defined(__OPEN64__)
R123_STATIC_INLINE uint64_t _mm_extract_lo64(__m128i si){
    union{
        uint64_t u64[2];
        __m128i m;
    }u;
    _mm_store_si128(&u.m, si);
    return u.u64[0];
}
#elif defined(__llvm__) || defined(__ICC)
R123_STATIC_INLINE uint64_t _mm_extract_lo64(__m128i si){
    return (uint64_t)_mm_cvtsi128_si64(si);
}
#else /* GNUC, others */
/* FWIW, gcc's emmintrin.h has had the 'x' spelling
   since at least gcc-3.4.4.  The no-'x' spelling showed up
   around 4.2. */
R123_STATIC_INLINE uint64_t _mm_extract_lo64(__m128i si){
    return (uint64_t)_mm_cvtsi128_si64x(si);
}
#endif
#if defined(__GNUC__) && __GNUC__ < 4
/* the cast builtins showed up in gcc4. */
R123_STATIC_INLINE __m128 _mm_castsi128_ps(__m128i si){
    return (__m128)si;
}
#endif

#ifdef __cplusplus

struct r123m128i{
    __m128i m;
#if R123_USE_CXX11_UNRESTRICTED_UNIONS
    // C++98 forbids a union member from having *any* constructors.
    // C++11 relaxes this, and allows union members to have constructors
    // as long as there is a "trivial" default construtor.  So in C++11
    // we can provide a r123m128i constructor with an __m128i argument, and still
    // have the default (and hence trivial) default constructor.
    r123m128i() = default;
    r123m128i(__m128i _m): m(_m){}
#endif
    r123m128i& operator=(const __m128i& rhs){ m=rhs; return *this;}
    r123m128i& operator=(R123_ULONG_LONG n){ m = _mm_set_epi64x(0, n); return *this;}
#if R123_USE_CXX11_EXPLICIT_CONVERSIONS
    // With C++0x we can attach explicit to the bool conversion operator
    // to disambiguate undesired promotions.  For g++, this works
    // only in 4.5 and above.
    explicit operator bool() const {return _bool();}
#else
    // Pre-C++0x, we have to do something else.  Google for the "safe bool"
    // idiom for other ideas...
    operator const void*() const{return _bool()?this:0;}
#endif
    operator __m128i() const {return m;}

private:
#if R123_USE_SSE4_1
    bool _bool() const{ return !_mm_testz_si128(m,m); }
#else
    bool _bool() const{ return 0xf != _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(m, _mm_setzero_si128()))); }
#endif
};

R123_STATIC_INLINE r123m128i& operator++(r123m128i& v){
    __m128i& c = v.m;
    __m128i zeroone = _mm_set_epi64x(R123_64BIT(0), R123_64BIT(1));
    c = _mm_add_epi64(c, zeroone);
    //return c;
#if R123_USE_SSE4_1
    __m128i zerofff = _mm_set_epi64x(0, ~(R123_64BIT(0)));
    if( R123_BUILTIN_EXPECT(_mm_testz_si128(c,zerofff), 0) ){
        __m128i onezero = _mm_set_epi64x(R123_64BIT(1), R123_64BIT(0));
        c = _mm_add_epi64(c, onezero);
    }
#else
    unsigned mask  = _mm_movemask_ps( _mm_castsi128_ps(_mm_cmpeq_epi32(c, _mm_setzero_si128())));
    // The low two bits of mask are 11 iff the low 64 bits of
    // c are zero.
    if( R123_BUILTIN_EXPECT((mask&0x3) == 0x3, 0) ){
        __m128i onezero = _mm_set_epi64x(1,0);
        c = _mm_add_epi64(c, onezero);
    }
#endif
    return v;
}

R123_STATIC_INLINE r123m128i& operator+=(r123m128i& lhs, R123_ULONG_LONG n){ 
    __m128i c = lhs.m;
    __m128i incr128 = _mm_set_epi64x(0, n);
    c = _mm_add_epi64(c, incr128);
    // return c;     // NO CARRY!  

    int64_t lo64 = _mm_extract_lo64(c);
    if((uint64_t)lo64 < n)
        c = _mm_add_epi64(c, _mm_set_epi64x(1,0));
    lhs.m = c;
    return lhs; 
}

// We need this one because it's present, but never used in r123array1xm128i::incr
R123_STATIC_INLINE bool operator<=(R123_ULONG_LONG, const r123m128i &){
    throw std::runtime_error("operator<=(unsigned long long, r123m128i) is unimplemented.");}

// The comparisons aren't implemented, but if we leave them out, and 
// somebody writes, e.g., M1 < M2, the compiler will do an implicit
// conversion through void*.  Sigh...
R123_STATIC_INLINE bool operator<(const r123m128i&, const r123m128i&){
    throw std::runtime_error("operator<(r123m128i, r123m128i) is unimplemented.");}
R123_STATIC_INLINE bool operator<=(const r123m128i&, const r123m128i&){
    throw std::runtime_error("operator<=(r123m128i, r123m128i) is unimplemented.");}
R123_STATIC_INLINE bool operator>(const r123m128i&, const r123m128i&){
    throw std::runtime_error("operator>(r123m128i, r123m128i) is unimplemented.");}
R123_STATIC_INLINE bool operator>=(const r123m128i&, const r123m128i&){
    throw std::runtime_error("operator>=(r123m128i, r123m128i) is unimplemented.");}

R123_STATIC_INLINE bool operator==(const r123m128i &lhs, const r123m128i &rhs){ 
    return 0xf==_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(lhs, rhs))); }
R123_STATIC_INLINE bool operator!=(const r123m128i &lhs, const r123m128i &rhs){ 
    return !(lhs==rhs);}
R123_STATIC_INLINE bool operator==(R123_ULONG_LONG lhs, const r123m128i &rhs){
    r123m128i LHS; LHS.m=_mm_set_epi64x(0, lhs); return LHS == rhs; }
R123_STATIC_INLINE bool operator!=(R123_ULONG_LONG lhs, const r123m128i &rhs){
    return !(lhs==rhs);}
R123_STATIC_INLINE std::ostream& operator<<(std::ostream& os, const r123m128i& m){
    union{
        uint64_t u64[2];
        __m128i m;
    }u;
    _mm_storeu_si128(&u.m, m.m);
    return os << u.u64[0] << " " << u.u64[1];
}

R123_STATIC_INLINE std::istream& operator>>(std::istream& is, r123m128i& m){
    uint64_t u64[2];
    is >> u64[0] >> u64[1];
    m.m = _mm_set_epi64x(u64[1], u64[0]);
    return is;
}

template<typename T> inline T assemble_from_u32(uint32_t *p32); // forward declaration

template <>
inline r123m128i assemble_from_u32<r123m128i>(uint32_t *p32){
    r123m128i ret;
    ret.m = _mm_set_epi32(p32[3], p32[2], p32[1], p32[0]);
    return ret;
}

#else

typedef struct {
    __m128i m;
} r123m128i;

#endif /* __cplusplus */

#else /* !R123_USE_SSE */
R123_STATIC_INLINE int haveAESNI(){
    return 0;
}
#endif /* R123_USE_SSE */

#endif /* _Random123_sse_dot_h__ */
