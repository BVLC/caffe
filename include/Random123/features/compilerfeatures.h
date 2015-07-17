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
/**

@page porting Preprocessor symbols for porting Random123 to different platforms.

The Random123 library is portable across C, C++, CUDA, OpenCL environments,
and multiple operating systems (Linux, Windows 7, Mac OS X, FreeBSD, Solaris).
This level of portability requires the abstraction of some features
and idioms that are either not standardized (e.g., asm statments), or for which 
different vendors have their own standards (e.g., SSE intrinsics) or for
which vendors simply refuse to conform to well-established standards (e.g., <inttypes.h>).

Random123/features/compilerfeatures.h
conditionally includes a compiler-or-OS-specific Random123/featires/XXXfeatures.h file which
defines appropriate values for the preprocessor symbols which can be used with
a specific compiler or OS.  Those symbols will then
be used by other header files and source files in the Random123
library (and may be used by applications) to control what actually
gets presented to the compiler.

Most of the symbols are boolean valued.  In general, they will
\b always be defined with value either 1 or 0, so do
\b NOT use \#ifdef.  Use \#if R123_USE_SOMETHING instead.

Library users can override any value by defining the pp-symbol with a compiler option,
e.g.,

    cc -DR123_USE_MULHILO64_C99 

will use a strictly c99 version of the full-width 64x64->128-bit multiplication
function, even if it would be disabled by default.

All boolean-valued pre-processor symbols in Random123/features/compilerfeatures.h start with the prefix R123_USE_
@verbatim
         AES_NI
         AES_OPENSSL
         SSE4_2
         SSE4_1
         SSE

         STD_RANDOM

         GNU_UINT128
         ASM_GNU
         ASM_MSASM

         CPUID_MSVC

         CXX11_RANDOM
         CXX11_TYPE_TRAITS
         CXX11_STATIC_ASSERT
         CXX11_CONSTEXPR
         CXX11_UNRESTRICTED_UNIONS
         CXX11_EXPLICIT_CONVERSIONS
         CXX11_LONG_LONG
         CXX11 
   
         X86INTRIN_H
         IA32INTRIN_H
         XMMINTRIN_H
         EMMINTRIN_H
         SMMINTRIN_H
         WMMINTRIN_H
         INTRIN_H

         MULHILO32_ASM
         MULHILO64_ASM
         MULHILO64_MSVC_INTRIN
         MULHILO64_CUDA_INTRIN
         MULHILO64_OPENCL_INTRIN
         MULHILO64_C99

         U01_DOUBLE
	 
@endverbatim
Most have obvious meanings.  Some non-obvious ones:

AES_NI and AES_OPENSSL are not mutually exclusive.  You can have one,
both or neither.

GNU_UINT128 says that it's safe to use __uint128_t, but it
does not require its use.  In particular, it should be
used in mulhilo<uint64_t> only if MULHILO64_ASM is unset.

If the XXXINTRIN_H macros are true, then one should
@code
#include <xxxintrin.h>
@endcode
to gain accesss to compiler intrinsics.

The CXX11_SOME_FEATURE macros allow the code to use specific
features of the C++11 language and library.  The catchall
In the absence of a specific CXX11_SOME_FEATURE, the feature
is controlled by the catch-all R123_USE_CXX11 macro.

U01_DOUBLE defaults on, and can be turned off (set to 0)
if one does not want the utility functions that convert to double
(i.e. u01_*_53()), e.g. on OpenCL without the cl_khr_fp64 extension.

There are a number of invariants that are always true.  Application code may
choose to rely on these:

<ul>
<li>ASM_GNU and ASM_MASM are mutually exclusive
<li>The "higher" SSE values imply the lower ones.
</ul>

There are also non-boolean valued symbols:

<ul>
<li>R123_STATIC_INLINE -
  According to both C99 and GNU99, the 'static inline' declaration allows
  the compiler to not emit code if the function is not used.  
  Note that the semantics of 'inline', 'static' and 'extern' in
  gcc have changed over time and are subject to modification by
  command line options, e.g., -std=gnu89, -fgnu-inline.
  Nevertheless, it appears that the meaning of 'static inline' 
  has not changed over time and (with a little luck) the use of 'static inline'
  here will be portable between versions of gcc and to other C99
  compilers.
  See: http://gcc.gnu.org/onlinedocs/gcc/Inline.html
       http://www.greenend.org.uk/rjk/2003/03/inline.html

<li>R123_FORCE_INLINE(decl) -
  which expands to 'decl', adorned with the compiler-specific
  embellishments to strongly encourage that the declared function be
  inlined.  If there is no such compiler-specific magic, it should
  expand to decl, unadorned.
   
<li>R123_CUDA_DEVICE - which expands to __device__ (or something else with
  sufficiently similar semantics) when CUDA is in use, and expands
  to nothing in other cases.

<li>R123_ASSERT(x) - which expands to assert(x), or maybe to nothing at
  all if we're in an environment so feature-poor that you can't even
  call assert (I'm looking at you, CUDA and OpenCL), or even include
  assert.h safely (OpenCL).

<li>R123_STATIC_ASSERT(expr,msg) - which expands to
  static_assert(expr,msg), or to an expression that
  will raise a compile-time exception if expr is not true.

<li>R123_ULONG_LONG - which expands to a declaration of the longest available
  unsigned integer.

<li>R123_64BIT(x) - expands to something equivalent to
  UINT64_C(x) from <stdint.h>, even in environments where <stdint.h>
  is not available, e.g., MSVC and OpenCL.

<li>R123_BUILTIN_EXPECT(expr,likely_value) - expands to something with
  the semantics of gcc's __builtin_expect(expr,likely_value).  If
  the environment has nothing like __builtin_expect, it should expand
  to just expr.
</ul>


\cond HIDDEN_FROM_DOXYGEN
*/

/* 
N.B.  When something is added to the list of features, it should be
added to each of the *features.h files, AND to examples/ut_features.cpp.
*/

/* N.B.  most other compilers (icc, nvcc, open64, llvm) will also define __GNUC__, so order matters. */
#if defined(__OPENCL_VERSION__) && __OPENCL_VERSION__ > 0
#include "openclfeatures.h"
#elif defined(__CUDACC__)
#include "nvccfeatures.h"
#elif defined(__ICC)
#include "iccfeatures.h"
#elif defined(__xlC__)
#include "xlcfeatures.h"
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#include "sunprofeatures.h"
#elif defined(__OPEN64__)
#include "open64features.h"
#elif defined(__clang__)
#include "clangfeatures.h"
#elif defined(__GNUC__)
#include "gccfeatures.h"
#elif defined(__PGI)
#include "pgccfeatures.h"
#elif defined(_MSC_FULL_VER)
#include "msvcfeatures.h"
#else
#error "Can't identify compiler.  You'll need to add a new xxfeatures.hpp"
{ /* maybe an unbalanced brace will terminate the compilation */
#endif

#ifndef R123_USE_CXX11
#define R123_USE_CXX11 (__cplusplus >= 201103L)
#endif

#ifndef R123_USE_CXX11_UNRESTRICTED_UNIONS
#define R123_USE_CXX11_UNRESTRICTED_UNIONS R123_USE_CXX11
#endif

#ifndef R123_USE_CXX11_STATIC_ASSERT
#define R123_USE_CXX11_STATIC_ASSERT R123_USE_CXX11
#endif

#ifndef R123_USE_CXX11_CONSTEXPR
#define R123_USE_CXX11_CONSTEXPR R123_USE_CXX11
#endif

#ifndef R123_USE_CXX11_EXPLICIT_CONVERSIONS
#define R123_USE_CXX11_EXPLICIT_CONVERSIONS R123_USE_CXX11
#endif

#ifndef R123_USE_CXX11_RANDOM
#define R123_USE_CXX11_RANDOM R123_USE_CXX11
#endif

#ifndef R123_USE_CXX11_TYPE_TRAITS
#define R123_USE_CXX11_TYPE_TRAITS R123_USE_CXX11
#endif

#ifndef R123_USE_CXX11_LONG_LONG
#define R123_USE_CXX11_LONG_LONG R123_USE_CXX11
#endif

#ifndef R123_USE_MULHILO64_C99
#define R123_USE_MULHILO64_C99 0
#endif

#ifndef R123_USE_MULHILO64_MULHI_INTRIN
#define R123_USE_MULHILO64_MULHI_INTRIN 0
#endif

#ifndef R123_USE_MULHILO32_MULHI_INTRIN
#define R123_USE_MULHILO32_MULHI_INTRIN 0
#endif

#ifndef R123_STATIC_ASSERT
#if R123_USE_CXX11_STATIC_ASSERT
#define R123_STATIC_ASSERT(expr, msg) static_assert(expr, msg)
#else
    /* if msg always_looked_like_this, we could paste it into the name.  Worth it? */
#define R123_STATIC_ASSERT(expr, msg) typedef char static_assertion[(!!(expr))*2-1]
#endif
#endif

#ifndef R123_CONSTEXPR
#if R123_USE_CXX11_CONSTEXPR
#define R123_CONSTEXPR constexpr
#else
#define R123_CONSTEXPR
#endif
#endif

#ifndef R123_USE_PHILOX_64BIT
#define R123_USE_PHILOX_64BIT (R123_USE_MULHILO64_ASM || R123_USE_MULHILO64_MSVC_INTRIN || R123_USE_MULHILO64_CUDA_INTRIN || R123_USE_GNU_UINT128 || R123_USE_MULHILO64_C99 || R123_USE_MULHILO64_OPENCL_INTRIN || R123_USE_MULHILO64_MULHI_INTRIN)
#endif

#ifndef R123_ULONG_LONG
#if defined(__cplusplus) && !R123_USE_CXX11_LONG_LONG
/* C++98 doesn't have long long.  It doesn't have uint64_t either, but
   we will have typedef'ed uint64_t to something in the xxxfeatures.h.
   With luck, it won't elicit complaints from -pedantic.  Cross your
   fingers... */
#define R123_ULONG_LONG uint64_t
#else
#define R123_ULONG_LONG unsigned long long
#endif
#endif

/* UINT64_C should have been #defined by XXXfeatures.h, either by
   #include <stdint.h> or through compiler-dependent hacks */
#ifndef R123_64BIT
#define R123_64BIT(x) UINT64_C(x)
#endif

#ifndef R123_THROW
#define R123_THROW(x)    throw (x)
#endif

/*
 * Windows.h (and perhaps other "well-meaning" code define min and
 * max, so there's a high chance that our definition of min, max
 * methods or use of std::numeric_limits min and max will cause
 * complaints in any program that happened to include Windows.h or
 * suchlike first.  We use the null macro below in our own header
 * files definition or use of min, max to defensively preclude
 * this problem.  It may not be enough; one might need to #define
 * NOMINMAX before including Windows.h or compile with -DNOMINMAX.
 */
#define R123_NO_MACRO_SUBST

/** \endcond */
