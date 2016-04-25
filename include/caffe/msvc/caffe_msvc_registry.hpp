#pragma once

#if defined(_MSC_VER)

#include "common.hpp"

#if !defined(GET_CLASS_GUARD_NAME)
#error "Cannot proceed without the macro GET_CLASS_GUARD_NAME"
#endif

#define FORCE_TO_LINK_CLASS(type) \
    extern char GET_CLASS_GUARD_NAME(type);   \
    static char __##type = GET_CLASS_GUARD_NAME(type);

namespace caffe {

// TEST-ADD-BEGIN
#ifdef CMAKE_BUILD
// TEST-ADD-END

    // LayerFactory is a special case 
    // that does not contain INSTANTIATE_CLASS.
    // Thus it is explicitly included here.
    FORCE_TO_LINK_CLASS(LayerFactory);

    // The content of the following file was generated 
    // by running the following command in cygwin:
    // $ grep -r "INSTANTIATE_CLASS" src/caffe/* | sort | cut -d "(" -f "2" | cut -d ")" -f 1 | awk '{ print "FORCE_TO_LINK_CLASS(", $0, ")" }'
    #include "caffe/msvc/caffe_classes.hpp"

// TEST-ADD-BEGIN
#endif  // #ifdef CMAKE_BUILD
// TEST-ADD-END

}

#endif

