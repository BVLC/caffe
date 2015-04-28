/*
    Copyright (c) 2011, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#ifdef _MSC_VER

#include <mmintrin.h>

__m128 operator*( __m128 a, __m128 b ){
  return _mm_mul_ps( a, b );
}
__m128 operator/( __m128 a, __m128 b ){
  return _mm_div_ps( a, b );
}
__m128 operator+( __m128 a, __m128 b ){
  return _mm_add_ps( a, b );
}
__m128 operator-( __m128 a, __m128 b ){
  return _mm_sub_ps( a, b );
}
__m128 operator*=( __m128 &a, __m128 b ){
  return a=_mm_mul_ps( a, b );
}
__m128 operator/=( __m128 &a, __m128 b ){
  return a=_mm_div_ps( a, b );
}
__m128 operator+=( __m128 &a, __m128 b ){
  return a=_mm_add_ps( a, b );
}
__m128 operator-=( __m128 &a, __m128 b ){
  return a=_mm_sub_ps( a, b );
}
__m128 operator-( __m128 a ){
  return _mm_sub_ps( _mm_set1_ps(0), a );
}

#endif //_MSC_VER
