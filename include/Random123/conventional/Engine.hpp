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
#ifndef __Engine_dot_hpp_
#define __Engine_dot_hpp_

#include "../features/compilerfeatures.h"
#include "../array.h"
#include <limits>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <vector>
#if R123_USE_CXX11_TYPE_TRAITS
#include <type_traits>
#endif

namespace r123{
/**
  If G satisfies the requirements of a CBRNG, and has a ctr_type whose
  value_type is an unsigned integral type, then Engine<G> satisfies
  the requirements of a C++0x "Uniform Random Number Engine" and can
  be used in any context where such an object is expected.

  Note that wrapping a counter based RNG with a traditional API in
  this way obscures much of the power of counter based PRNGs.
  Nevertheless, it may be of value in applications that are already
  coded to work with the C++0x random number engines.

  The MicroURNG template in MicroURNG.hpp
  provides the more limited functionality of a C++0x "Uniform
  Random Number Generator", but leaves the application in control
  of counters and keys and hence may be preferable to the Engine template.
  For example, a MicroURNG allows one to use C++0x "Random Number
  Distributions"  without giving up control over the counters
  and keys.
*/ 

template<typename CBRNG>
struct Engine {
    typedef CBRNG cbrng_type;
    typedef typename CBRNG::ctr_type ctr_type;
    typedef typename CBRNG::key_type key_type;
    typedef typename CBRNG::ukey_type ukey_type;
    typedef typename ctr_type::value_type result_type;
    typedef size_t elem_type;

protected:
    cbrng_type b;
    key_type key;
    ukey_type ukey;
    ctr_type c;
    elem_type elem;
    ctr_type v;

    void fix_invariant(){
        if( elem != 0 ) {
            v = b(c, key);
	}
    }        
public:
    explicit Engine() : b(), c(), elem() {
	ukey_type x = {{}};
	ukey = x;
        key = ukey;
    }
    explicit Engine(result_type r) : b(), c(), elem() {
        ukey_type x = {{typename ukey_type::value_type(r)}};
        ukey = x;
        key = ukey;
    }
    // 26.5.3 says that the SeedSeq templates shouldn't particpate in
    // overload resolution unless the type qualifies as a SeedSeq.
    // How that is determined is unspecified, except that "as a
    // minimum a type shall not qualify as a SeedSeq if it is
    // implicitly convertible to a result_type."  
    //
    // First, we make sure that even the non-const copy constructor
    // works as expected.  In addition, if we've got C++0x
    // type_traits, we use enable_if and is_convertible to implement
    // the convertible-to-result_type restriction.  Otherwise, the
    // template is unconditional and will match in some surpirsing
    // and undesirable situations.
    Engine(Engine& e) : b(e.b), ukey(e.ukey), c(e.c), elem(e.elem){
        key = ukey;
        fix_invariant();
    }
    Engine(const Engine& e) : b(e.b), ukey(e.ukey), c(e.c), elem(e.elem){
        key = ukey;
        fix_invariant();
    }

    template <typename SeedSeq>
    explicit Engine(SeedSeq &s
#if R123_USE_CXX11_TYPE_TRAITS
                    , typename std::enable_if<!std::is_convertible<SeedSeq, result_type>::value>::type* =0
#endif
                    )
        : b(), c(), elem() {
        ukey = ukey_type::seed(s);
        key = ukey;
    }
    void seed(result_type r){
        *this = Engine(r);
    }
    template <typename SeedSeq>
    void seed(SeedSeq &s
#if R123_USE_CXX11_TYPE_TRAITS
                    , typename std::enable_if<!std::is_convertible<SeedSeq, result_type>::value>::type* =0
#endif
              ){ 
        *this = Engine(s);
    }
    void seed(){
        *this = Engine();
    }
    friend bool operator==(const Engine& lhs, const Engine& rhs){
        return lhs.c==rhs.c && lhs.elem == rhs.elem && lhs.ukey == rhs.ukey;
    }
    friend bool operator!=(const Engine& lhs, const Engine& rhs){
        return lhs.c!=rhs.c || lhs.elem != rhs.elem || lhs.ukey!=rhs.ukey;
    }

    friend std::ostream& operator<<(std::ostream& os, const Engine& be){
        return os << be.c << " " << be.ukey << " " << be.elem;
    }

    friend std::istream& operator>>(std::istream& is, Engine& be){
        is >> be.c >> be.ukey >> be.elem;
        be.key = be.ukey;
        be.fix_invariant();
        return is;
    }

    // The <random> shipped with MacOS Xcode 4.5.2 imposes a
    // non-standard requirement that URNGs also have static data
    // members: _Min and _Max.  Later versions of libc++ impose the
    // requirement only when constexpr isn't supported.  Although the
    // Xcode 4.5.2 requirement is clearly non-standard, it is unlikely
    // to be fixed and it is very easy work around.  We certainly
    // don't want to go to great lengths to accommodate every buggy
    // library we come across, but in this particular case, the effort
    // is low and the benefit is high, so it's worth doing.  Thanks to
    // Yan Zhou for pointing this out to us.  See similar code in
    // ../MicroURNG.hpp
    const static result_type _Min = 0;
    const static result_type _Max = ~((result_type)0);

    static R123_CONSTEXPR result_type min R123_NO_MACRO_SUBST () { return _Min; }
    static R123_CONSTEXPR result_type max R123_NO_MACRO_SUBST () { return _Max; }

    result_type operator()(){
        if( c.size() == 1 )     // short-circuit the scalar case.  Compilers aren't mind-readers.
            return b(c.incr(), key)[0];
        if( elem == 0 ){
            v = b(c.incr(), key);
            elem = c.size();
        }
        return v[--elem];
    }

    void discard(R123_ULONG_LONG skip){
        // don't forget:  elem counts down
        size_t nelem = c.size();
	size_t sub = skip % nelem;
        skip /= nelem;
	if (elem < sub) {
	    elem += nelem;
	    skip++;
	}
	elem -= sub;
        c.incr(skip);
        fix_invariant();
    }
         
    //--------------------------
    // Some bonus methods, not required for a Random Number
    // Engine

    // Constructors and seed() method for ukey_type seem useful
    // We need const and non-const to supersede the SeedSeq template.
    explicit Engine(const ukey_type &uk) : key(uk), ukey(uk), c(), elem(){}
    explicit Engine(ukey_type &uk) : key(uk), ukey(uk), c(), elem(){}
    void seed(const ukey_type& uk){
        *this = Engine(uk);
    }        
    void seed(ukey_type& uk){
        *this = Engine(uk);
    }        

    // Forward the e(counter) to the CBRNG we are templated
    // on, using the current value of the key.
    ctr_type operator()(const ctr_type& c) const{
        return b(c, key);
    }

    // Since you can seed *this with a ukey_type, it seems reasonable
    // to allow the caller to know what seed/ukey *this is using.
    ukey_type getseed() const{
        return ukey;
    }

    // Maybe the caller want's to know the details of
    // the internal state, e.g., so it can call a different
    // bijection with the same counter.
    std::pair<ctr_type, elem_type> getcounter() const {
        return make_pair(c,  elem);
    }

    // And the inverse.
    void setcounter(const ctr_type& _c, elem_type _elem){
        static const size_t nelem = c.size();
        if( elem > nelem )
            throw std::range_error("Engine::setcounter called  with elem out of range");
        c = _c;
        elem = _elem;
        fix_invariant();
    }
};
} // namespace r123

#endif
