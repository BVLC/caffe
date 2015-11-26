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
#ifndef __clangfeatures_dot_hpp
#define __clangfeatures_dot_hpp

#ifndef R123_USE_X86INTRIN_H
#define R123_USE_X86INTRIN_H ((defined(__x86_64__)||defined(__i386__)))
#endif

#ifndef R123_USE_CXX11_UNRESTRICTED_UNIONS
#define R123_USE_CXX11_UNRESTRICTED_UNIONS __has_feature(cxx_unrestricted_unions)
#endif

#ifndef R123_USE_CXX11_STATIC_ASSERT
#define R123_USE_CXX11_STATIC_ASSERT __has_feature(cxx_static_assert)
#endif

#ifndef R123_USE_CXX11_CONSTEXPR
#define R123_USE_CXX11_CONSTEXPR __has_feature(cxx_constexpr)
#endif

#ifndef R123_USE_CXX11_EXPLICIT_CONVERSIONS
#define R123_USE_CXX11_EXPLICIT_CONVERSIONS __has_feature(cxx_explicit_conversions)
#endif

// With clang-3.0, the apparently simpler:
//  #define R123_USE_CXX11_RANDOM __has_include(<random>)
// dumps core.
#ifndef R123_USE_CXX11_RANDOM
#if __cplusplus>=201103L && __has_include(<random>)
#define R123_USE_CXX11_RANDOM 1
#else
#define R123_USE_CXX11_RANDOM 0
#endif
#endif

#ifndef R123_USE_CXX11_TYPE_TRAITS
#if __cplusplus>=201103L && __has_include(<type_traits>)
#define R123_USE_CXX11_TYPE_TRAITS 1
#else
#define R123_USE_CXX11_TYPE_TRAITS 0
#endif
#endif

#include "gccfeatures.h"

#endif
