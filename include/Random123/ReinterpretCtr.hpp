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
#ifndef __ReinterpretCtr_dot_hpp__
#define __ReinterpretCtr_dot_hpp__

#include "features/compilerfeatures.h"
#include <cstring>

namespace r123{
/*!
  ReinterpretCtr uses memcpy to map back and forth
  between a CBRNG's ctr_type and the specified ToType.  For example,
  after:

    typedef ReinterpretCtr<r123array4x32, Philox2x64> G;

  G is a bona fide CBRNG with ctr_type r123array4x32.

  WARNING:  ReinterpretCtr is endian dependent.  The
  values returned by G, declared as above,
  will depend on the endianness of the machine on which it runs.
 */

template <typename ToType, typename CBRNG>
struct ReinterpretCtr{
    typedef ToType ctr_type;
    typedef typename CBRNG::key_type key_type;
    typedef typename CBRNG::ctr_type bctype;
    typedef typename CBRNG::ukey_type ukey_type;
    R123_STATIC_ASSERT(sizeof(ToType) == sizeof(bctype) && sizeof(typename bctype::value_type) != 16, 
                       "ReinterpretCtr:  sizeof(ToType) is not the same as sizeof(CBRNG::ctr_type) or CBRNG::ctr_type::value_type looks like it might be __m128i");
    // It's amazingly difficult to safely do conversions with __m128i.
    // If we use the operator() implementation below with a CBRNG
    // whose ctr_type is r123array1xm128i, gcc4.6 optimizes away the
    // memcpys, inlines the operator()(c,k), and produces assembly
    // language that ends with an aesenclast instruction with a
    // destination operand pointing to an unaligned memory address ...
    // Segfault!  See:  http://gcc.gnu.org/bugzilla/show_bug.cgi?id=50444
    // MSVC also produces code that crashes.  We suspect a
    // similar mechanism but haven't done the debugging necessary to
    // be sure.  We were able to 'fix' gcc4.6 by making bc a mutable
    // data member rather than declaring it in the scope of
    // operator().  That didn't fix the MSVC problems, though.
    //
    // Conclusion - don't touch __m128i, at least for now.  The
    // easiest (but highly imprecise) way to do that is the static
    // assertion above that rejects bctype::value_types of size 16. -
    // Sep 2011.
    ctr_type  operator()(ctr_type c, key_type k){
        bctype bc;
        std::memcpy(&bc, &c, sizeof(c));
        CBRNG b;
        bc = b(bc, k);
        std::memcpy(&c, &bc, sizeof(bc));
        return c;
    }
};
} // namespace r123
#endif
