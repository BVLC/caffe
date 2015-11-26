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
#ifndef __openclfeatures_dot_hpp
#define __openclfeatures_dot_hpp

#ifndef R123_STATIC_INLINE
#define R123_STATIC_INLINE inline
#endif

#ifndef R123_FORCE_INLINE
#define R123_FORCE_INLINE(decl) decl __attribute__((always_inline))
#endif

#ifndef R123_CUDA_DEVICE
#define R123_CUDA_DEVICE
#endif

#ifndef R123_ASSERT
#define R123_ASSERT(x)
#endif

#ifndef R123_BUILTIN_EXPECT
#define R123_BUILTIN_EXPECT(expr,likely) expr
#endif

#ifndef R123_USE_GNU_UINT128
#define R123_USE_GNU_UINT128 0
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
#define R123_USE_MULHILO64_OPENCL_INTRIN 1
#endif

#ifndef R123_USE_AES_NI
#define R123_USE_AES_NI 0
#endif

// XXX ATI APP SDK 2.4 clBuildProgram SEGVs if one uses uint64_t instead of
// ulong to mul_hi.  And gets lots of complaints from stdint.h
// on some machines.
// But these typedefs mean we cannot include stdint.h with
// these headers?  Do we need R123_64T, R123_32T, R123_8T?
typedef ulong uint64_t;
typedef uint  uint32_t;
typedef uchar uint8_t;
#define UINT64_C(x) ((ulong)(x##UL))

#endif
