/*
 * This class is modified from Philipp Krähenbühl's NIPS 2011 code
 * See his webstire for more information
 * http://graphics.stanford.edu/projects/densecrf/
 *
 * Liang-Chieh Chen, 2015
*/ 
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

#ifndef _PERMUTOHEDRAL_H
#define _PERMUTOHEDRAL_H

#include <cstdlib>

#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>

#ifdef __SSE__
// SSE Permutohedral lattice
# define SSE_PERMUTOHEDRAL
#endif

#if defined(SSE_PERMUTOHEDRAL)
# include <emmintrin.h>
# include <xmmintrin.h>
# ifdef __SSE4_1__
#  include <smmintrin.h>
# endif
#endif



/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/

class Permutohedral {
 protected:
  int * offset_;
  float * barycentric_;
  
  struct Neighbors{
    int n1, n2;
    Neighbors(int n1=0, int n2=0 ) : n1(n1),n2(n2) {}
  };
  Neighbors * blur_neighbors_;
  // Number of elements, size of sparse discretized space, dimension of features
  int N_, M_, d_;
 public:
  Permutohedral();
  virtual ~Permutohedral();

  void init(const float* feature, int feature_size, int N);

#ifdef SSE_PERMUTOHEDRAL
  void compute(__m128* out, const __m128* in, int value_size, int in_offset = 0, int out_offset = 0, int in_size = -1, int out_size = -1) const;
 #endif
 
  void compute(float* out, const float* in, int value_size, int in_offset = 0, int out_offset = 0, int in_size = -1, int out_size = -1) const;

};


#endif
