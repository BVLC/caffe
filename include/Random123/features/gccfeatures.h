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
#ifndef __gccfeatures_dot_hpp
#define __gccfeatures_dot_hpp

#define R123_GNUC_VERSION (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)

#if !defined(__x86_64__) && !defined(__i386__) && !defined(__powerpc__)
#  error "This code has only been tested on x86 and powerpc platforms."
#include <including_a_nonexistent_file_will_stop_some_compilers_from_continuing_with_a_hopeless_task>
{ /* maybe an unbalanced brace will terminate the compilation */
 /* Feel free to try the Random123 library on other architectures by changing
 the conditions that reach this error, but you should consider it a
 porting exercise and expect to encounter bugs and deficiencies.
 Please let the authors know of any successes (or failures). */
#endif

#ifdef __powerpc__
#include <ppu_intrinsics.h>
#endif

#ifndef R123_STATIC_INLINE
#define R123_STATIC_INLINE static __inline__
#endif

#ifndef R123_FORCE_INLINE
#if R123_GNUC_VERSION >= 40000
#define R123_FORCE_INLINE(decl) decl __attribute__((always_inline))
#else
#define R123_FORCE_INLINE(decl) decl
#endif
#endif

#ifndef R123_CUDA_DEVICE
#define R123_CUDA_DEVICE
#endif

#ifndef R123_ASSERT
#include <assert.h>
#define R123_ASSERT(x) assert(x)
#endif

#ifndef R123_BUILTIN_EXPECT
#define R123_BUILTIN_EXPECT(expr,likely) __builtin_expect(expr,likely)
#endif

/* According to the C++0x standard, we should be able to test the numeric
   value of __cplusplus == 199701L for C++98, __cplusplus == 201103L for C++0x
   But gcc has had an open bug  http://gcc.gnu.org/bugzilla/show_bug.cgi?id=1773
   since early 2001, which was finally fixed in 4.7 (early 2012).  For
   earlier versions, the only way  to detect whether --std=c++0x was requested
   on the command line is to look at the __GCC_EXPERIMENTAL_CXX0X__ pp-symbol.
*/
#define GNU_CXX11 (__cplusplus>=201103L || (R123_GNUC_VERSION<40700 && defined(__GCC_EXPERIMENTAL_CXX0X__) ))

#ifndef R123_USE_CXX11_UNRESTRICTED_UNIONS
#define R123_USE_CXX11_UNRESTRICTED_UNIONS ((R123_GNUC_VERSION >= 40600) && GNU_CXX11)
#endif

#ifndef R123_USE_CXX11_STATIC_ASSERT
#define R123_USE_CXX11_STATIC_ASSERT ((R123_GNUC_VERSION >= 40300) && GNU_CXX11)
#endif

#ifndef R123_USE_CXX11_CONSTEXPR
#define R123_USE_CXX11_CONSTEXPR ((R123_GNUC_VERSION >= 40600) && GNU_CXX11)
#endif

#ifndef R123_USE_CXX11_EXPLICIT_CONVERSIONS
#define R123_USE_CXX11_EXPLICIT_CONVERSIONS ((R123_GNUC_VERSION >= 40500) && GNU_CXX11)
#endif

#ifndef R123_USE_CXX11_RANDOM
#define R123_USE_CXX11_RANDOM ((R123_GNUC_VERSION>=40500) && GNU_CXX11)
#endif

#ifndef R123_USE_CXX11_TYPE_TRAITS
#define R123_USE_CXX11_TYPE_TRAITS ((R123_GNUC_VERSION>=40400) && GNU_CXX11)
#endif

#ifndef R123_USE_AES_NI
#ifdef __AES__
#define R123_USE_AES_NI 1
#else
#define R123_USE_AES_NI 0
#endif
#endif

#ifndef R123_USE_SSE4_2
#ifdef __SSE4_2__
#define R123_USE_SSE4_2 1
#else
#define R123_USE_SSE4_2 0
#endif
#endif

#ifndef R123_USE_SSE4_1
#ifdef __SSE4_1__
#define R123_USE_SSE4_1 1
#else
#define R123_USE_SSE4_1 0
#endif
#endif

#ifndef R123_USE_SSE
/* There's no point in trying to compile SSE code in Random123
   unless SSE2 is available. */
#ifdef __SSE2__
#define R123_USE_SSE 1
#else
#define R123_USE_SSE 0
#endif
#endif

#ifndef R123_USE_AES_OPENSSL
/* There isn't really a good way to tell at compile time whether
   openssl is available.  Without a pre-compilation configure-like
   tool, it's less error-prone to guess that it isn't available.  Add
   -DR123_USE_AES_OPENSSL=1 and any necessary LDFLAGS or LDLIBS to
   play with openssl */
#define R123_USE_AES_OPENSSL 0
#endif

#ifndef R123_USE_GNU_UINT128
#ifdef __x86_64__
#define R123_USE_GNU_UINT128 1
#else
#define R123_USE_GNU_UINT128 0
#endif
#endif

#ifndef R123_USE_ASM_GNU
#define R123_USE_ASM_GNU (defined(__x86_64__)||defined(__i386__))
#endif

#ifndef R123_USE_CPUID_MSVC
#define R123_USE_CPUID_MSVC 0
#endif

#ifndef R123_USE_X86INTRIN_H
#define R123_USE_X86INTRIN_H ((defined(__x86_64__)||defined(__i386__)) && R123_GNUC_VERSION >= 40402)
#endif

#ifndef R123_USE_IA32INTRIN_H
#define R123_USE_IA32INTRIN_H 0
#endif

#ifndef R123_USE_XMMINTRIN_H
#define R123_USE_XMMINTRIN_H 0
#endif

#ifndef R123_USE_EMMINTRIN_H
/* gcc -m64 on Solaris 10 defines __SSE2__ but doesn't have 
   emmintrin.h in the include search path.  This is
   so broken that I refuse to try to work around it.  If this
   affects you, figure out where your emmintrin.h lives and
   add an appropriate -I to your CPPFLAGS.  Or add -DR123_USE_SSE=0. */
#define R123_USE_EMMINTRIN_H (R123_USE_SSE && (R123_GNUC_VERSION < 40402))
#endif

#ifndef R123_USE_SMMINTRIN_H
#define R123_USE_SMMINTRIN_H ((R123_USE_SSE4_1 || R123_USE_SSE4_2) && (R123_GNUC_VERSION < 40402))
#endif

#ifndef R123_USE_WMMINTRIN_H
#define R123_USE_WMMINTRIN_H 0
#endif

#ifndef R123_USE_INTRIN_H
#define R123_USE_INTRIN_H 0
#endif

#ifndef R123_USE_MULHILO32_ASM
#define R123_USE_MULHILO32_ASM 0
#endif

#ifndef R123_USE_MULHILO64_ASM
#define R123_USE_MULHILO64_ASM 0
#endif

#ifndef R123_USE_MULHILO64_MSVC_INTRIN
#define R123_USE_MULHILO64_MSVC_INTRIN 0
#endif

#ifndef R123_USE_MULHILO64_CUDA_INTRIN
#define R123_USE_MULHILO64_CUDA_INTRIN 0
#endif

#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
#define R123_USE_MULHILO64_OPENCL_INTRIN 0
#endif

#ifndef R123_USE_MULHILO64_MULHI_INTRIN
#define R123_USE_MULHILO64_MULHI_INTRIN (defined(__powerpc64__))
#endif

#ifndef R123_MULHILO64_MULHI_INTRIN
#define R123_MULHILO64_MULHI_INTRIN __mulhdu
#endif

#ifndef R123_USE_MULHILO32_MULHI_INTRIN
#define R123_USE_MULHILO32_MULHI_INTRIN 0
#endif

#ifndef R123_MULHILO32_MULHI_INTRIN
#define R123_MULHILO32_MULHI_INTRIN __mulhwu
#endif

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#include <stdint.h>
#ifndef UINT64_C
#error UINT64_C not defined.  You must define __STDC_CONSTANT_MACROS before you #include <stdint.h>
#endif

/* If you add something, it must go in all the other XXfeatures.hpp
   and in ../ut_features.cpp */
#endif
