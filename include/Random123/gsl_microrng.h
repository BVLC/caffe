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
#ifndef __r123_gslmicrorng_dot_h__
#define __r123_gslmicrorng_dot_h__


#include <gsl/gsl_rng.h>
#include <string.h>

/**   The macro: GSL_MICRORNG(NAME, CBRNGNAME) is the GSL
   analog analog of the C++ r123::MicroURNG template.  It declares a gsl_rng
   type named gsl_rng_NAME which uses the underlying CBRNGNAME
   and can be invoked a limited number of times between calls to NAME_reset.

   When the underlying CBRNG's \c ctr_t is an \ref arrayNxW "r123arrayNxW",
   and the gsl_rng_NAME may called up to \c N*2^32 times 
   between calls to \c NAME_reset.

   \c NAME_reset takes a gsl_rng_NAME type, a counter and a key as arguments.
   It restarts the micro-rng with a new base counter and key.

   Note that you must call NAME_reset before the first use
   of a gsl_rng.  NAME_reset is not called automatically by
   gsl_rng_alloc().

   @code
   #include <Random123/threefry.h>
   #include <Random123/gsl_microrng.h> // this file
   GSL_MICRORNG(microcbrng, threefry4x64, 20)	// creates gsl_rng_microcbrng

   int main(int argc, char** argv) {
	gsl_rng *r = gsl_rng_alloc(gsl_rng_microcbrng);
	threefry4x64_ctr_t c = {{}};
	threefry4x64_key_t k = {{}};

	for (...) {
	    c.v[0] = ??; //  some application variable
	    microcbrng_reset(r, c, k);
	    for (...) {
		// gaussian calls r several times.  It is safe for
		// r to be used upto 2^20 times in this loop
		something[i] = gsl_ran_gaussian(r, 1.5);
	    }
	}
   }
   @endcode
   
*/

#define GSL_MICRORNG(NAME, CBRNGNAME)                                   \
const gsl_rng_type *gsl_rng_##NAME;                                     \
                                                                        \
typedef struct{                                                         \
    CBRNGNAME##_ctr_t ctr;                                              \
    CBRNGNAME##_ctr_t r;                                                \
    CBRNGNAME##_key_t key;                                              \
    R123_ULONG_LONG n;                                                  \
    int elem;                                                           \
} NAME##_state;                                                         \
                                                                        \
static unsigned long int NAME##_get(void *vstate){                      \
    NAME##_state *st = (NAME##_state *)vstate;                          \
    const int N=sizeof(st->ctr.v)/sizeof(st->ctr.v[0]);                 \
    if( st->elem == 0 ){                                                \
        CBRNGNAME##_ctr_t c = st->ctr;                                  \
        c.v[N-1] |= st->n<<(R123_W(CBRNGNAME##_ctr_t)-32);              \
        st->n++;                                                        \
        st->r = CBRNGNAME(c, st->key);                                  \
        st->elem = N;                                                   \
    }                                                                   \
    return 0xffffffff & st->r.v[--st->elem];                            \
}                                                                       \
                                                                        \
static double                                                           \
NAME##_get_double (void * vstate)                                       \
{                                                                       \
    return NAME##_get (vstate)/4294967296.;                             \
}                                                                       \
                                                                        \
static void NAME##_set(void *vstate, unsigned long int s){              \
    NAME##_state *st = (NAME##_state *)vstate;                          \
    (void)s; /* ignored */                                              \
    st->elem = 0;                                                       \
    st->n = ~0; /* will abort if _reset is not called */                \
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
R123_STATIC_INLINE void NAME##_reset(const gsl_rng* gr, CBRNGNAME##_ctr_t c, CBRNGNAME##_key_t k) { \
    NAME##_state* state = (NAME##_state *)gr->state;                    \
    state->ctr = c;                                                     \
    state->key = k;                                                     \
    state->n = 0;                                                       \
    state->elem = 0;                                                    \
}                                                                       \
                                                                        \
const gsl_rng_type *gsl_rng_##NAME = &NAME##_type

#endif
