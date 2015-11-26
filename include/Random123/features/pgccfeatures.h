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

Copyright (c) 2013, Los Alamos National Security, LLC
All rights reserved.

Copyright 2013. Los Alamos National Security, LLC. This software was produced
under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National
Laboratory (LANL), which is operated by Los Alamos National Security, LLC for
the U.S. Department of Energy. The U.S. Government has rights to use,
reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS
ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified
to produce derivative works, such modified software should be clearly marked,
so as not to confuse it with the version available from LANL.
*/
#ifndef __pgccfeatures_dot_hpp
#define __pgccfeatures_dot_hpp

#if !defined(__x86_64__) && !defined(__i386__)
#  error "This code has only been tested on x86 platforms."
#include <including_a_nonexistent_file_will_stop_some_compilers_from_continuing_with_a_hopeless_task>
{ /* maybe an unbalanced brace will terminate the compilation */
 /* Feel free to try the Random123 library on other architectures by changing
 the conditions that reach this error, but you should consider it a
 porting exercise and expect to encounter bugs and deficiencies.
 Please let the authors know of any successes (or failures). */
#endif

#ifndef R123_STATIC_INLINE
#define R123_STATIC_INLINE static inline
#endif

/* Found this example in PGI's emmintrin.h. */
#ifndef R123_FORCE_INLINE
#define R123_FORCE_INLINE(decl) decl __attribute__((__always_inline__))
#endif

#ifndef R123_CUDA_DEVICE
#define R123_CUDA_DEVICE
#endif

#ifndef R123_ASSERT
#include <assert.h>
#define R123_ASSERT(x) assert(x)
#endif

#ifndef R123_BUILTIN_EXPECT
#define R123_BUILTIN_EXPECT(expr,likely) (expr)
#endif

/* PGI through 13.2 doesn't appear to support AES-NI. */
#ifndef R123_USE_AES_NI
#define R123_USE_AES_NI 0
#endif

/* PGI through 13.2 appears to support MMX, SSE, SSE3, SSE3, SSSE3, SSE4a, and
   ABM, but not SSE4.1 or SSE4.2. */
#ifndef R123_USE_SSE4_2
#define R123_USE_SSE4_2 0
#endif

#ifndef R123_USE_SSE4_1
#define R123_USE_SSE4_1 0
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
#define R123_USE_GNU_UINT128 0
#endif

#ifndef R123_USE_ASM_GNU
#define R123_USE_ASM_GNU 1
#endif

#ifndef R123_USE_CPUID_MSVC
#define R123_USE_CPUID_MSVC 0
#endif

#ifndef R123_USE_X86INTRIN_H
#define R123_USE_X86INTRIN_H 0
#endif

#ifndef R123_USE_IA32INTRIN_H
#define R123_USE_IA32INTRIN_H 0
#endif

/* emmintrin.h from PGI #includes xmmintrin.h but then complains at link time
   about undefined references to _mm_castsi128_ps(__m128i).  Why? */
#ifndef R123_USE_XMMINTRIN_H
#define R123_USE_XMMINTRIN_H 1
#endif

#ifndef R123_USE_EMMINTRIN_H
#define R123_USE_EMMINTRIN_H 1
#endif

#ifndef R123_USE_SMMINTRIN_H
#define R123_USE_SMMINTRIN_H 0
#endif

#ifndef R123_USE_WMMINTRIN_H
#define R123_USE_WMMINTRIN_H 0
#endif

#ifndef R123_USE_INTRIN_H
#ifdef __ABM__
#define R123_USE_INTRIN_H 1
#else
#define R123_USE_INTRIN_H 0
#endif
#endif

#ifndef R123_USE_MULHILO32_ASM
#define R123_USE_MULHILO32_ASM 0
#endif

#ifndef R123_USE_MULHILO64_MULHI_INTRIN
#define R123_USE_MULHILO64_MULHI_INTRIN 0
#endif

#ifndef R123_USE_MULHILO64_ASM
#define R123_USE_MULHILO64_ASM 1
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
