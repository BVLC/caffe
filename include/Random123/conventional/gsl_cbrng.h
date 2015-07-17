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
#ifndef __r123_compat_gslrng_dot_h__
#define __r123_compat_gslrng_dot_h__

#include <gsl/gsl_rng.h>
#include <string.h>

/**
   The macro:  GSL_CBRNG(NAME, CBRNGNAME)
   declares the necessary structs and  constants that define a
   gsl_rng_NAME type based on the counter-based RNG CBRNGNAME.  For example:

   Usage:

   @code
   #include <Random123/threefry.h>
   #include <Random123/conventional/gsl_cbrng.h>  // this file
   GSL_CBRNG(cbrng, threefry4x32); // creates gsl_rng_cbrng

   int main(int argc, char **argv){
       gsl_rng *r = gsl_rng_alloc(gsl_rng_cbrng);
       ... use r as you would use any other gsl_rng ...   
    }
    @endcode

    It requires that NAME be the name of a CBRNG that follows the
    naming and stylistic conventions of the Random123 library.

    Note that wrapping a \ref CBRNG "counter-based PRNG" with a traditional API in
    this way obscures much of the power of the CBRNG API.
    Nevertheless, it may be of value to applications that are already
    coded to work with GSL random number generators, and that wish
    to use the RNGs in the Random123 library.

 */ 

#define GSL_CBRNG(NAME, CBRNGNAME)                                      \
const gsl_rng_type *gsl_rng_##NAME;                                     \
                                                                        \
typedef struct{                                                         \
    CBRNGNAME##_ctr_t ctr;                                                   \
    CBRNGNAME##_ctr_t r;                                                     \
    CBRNGNAME##_key_t key;                                                   \
    int elem;                                                           \
} NAME##_state;                                                         \
                                                                        \
static unsigned long int NAME##_get(void *vstate){                      \
    NAME##_state *st = (NAME##_state *)vstate;                          \
    const int N=sizeof(st->ctr.v)/sizeof(st->ctr.v[0]);                 \
    if( st->elem == 0 ){                                                \
        ++st->ctr.v[0];                                                 \
        if( N>1 && st->ctr.v[0] == 0 ) ++st->ctr.v[1];                  \
        if( N>2 && st->ctr.v[1] == 0 ) ++st->ctr.v[2];                  \
        if( N>3 && st->ctr.v[2] == 0 ) ++st->ctr.v[3];                  \
        st->r = CBRNGNAME(st->ctr, st->key);                                 \
        st->elem = N;                                                   \
    }                                                                   \
    return 0xffffffffUL & st->r.v[--st->elem];                          \
}                                                                       \
                                                                        \
static double                                                           \
NAME##_get_double (void * vstate)                                       \
{                                                                       \
    return NAME##_get (vstate)/4294967296.0;                            \
}                                                                       \
                                                                        \
static void NAME##_set(void *vstate, unsigned long int s){              \
    NAME##_state *st = (NAME##_state *)vstate;                          \
    st->elem = 0;                                                       \
    /* Assume that key and ctr have an array member, v,                 \
       as if they are r123arrayNxW.  If not, this will fail             \
       to compile.  In particular, this macro fails to compile          \
       when the underlying CBRNG requires use of keyinit */             \
    memset(&st->ctr.v[0], 0, sizeof(st->ctr.v));                        \
    memset(&st->key.v[0], 0, sizeof(st->key.v));                        \
    /* GSL 1.15 documentation says this about gsl_rng_set:              \
         Note that the most generators only accept 32-bit seeds, with higher \
         values being reduced modulo 2^32.  For generators with smaller \
         ranges the maximum seed value will typically be lower.         \
     so we won't jump through any hoops here to deal with               \
     high bits if sizeof(unsigned long) > sizeof(uint32_t). */          \
    st->key.v[0] = s;                                                   \
}                                                                       \
                                                                        \
static const gsl_rng_type NAME##_type = {                               \
    #NAME,                                                              \
    0xffffffffUL,                                                       \
    0,                                                                  \
    sizeof(NAME##_state),                                               \
    &NAME##_set,                                                        \
    &NAME##_get,                                                        \
    &NAME##_get_double                                                  \
};                                                                      \
                                                                        \
const gsl_rng_type *gsl_rng_##NAME = &NAME##_type

#endif

