/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * Thread utilities for reading memory using PTX cache modifiers.
 */

#pragma once

#include <cuda.h>

#include <iterator>

#include "../util_ptx.cuh"
#include "../util_type.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup UtilIo
 * @{
 */

//-----------------------------------------------------------------------------
// Tags and constants
//-----------------------------------------------------------------------------

/**
 * \brief Enumeration of cache modifiers for memory load operations.
 */
enum CacheLoadModifier
{
    LOAD_DEFAULT,       ///< Default (no modifier)
    LOAD_CA,            ///< Cache at all levels
    LOAD_CG,            ///< Cache at global level
    LOAD_CS,            ///< Cache streaming (likely to be accessed once)
    LOAD_CV,            ///< Cache as volatile (including cached system lines)
    LOAD_LDG,           ///< Cache as texture
    LOAD_VOLATILE,      ///< Volatile (any memory space)
};


/**
 * \name Thread I/O (cache modified)
 * @{
 */

/**
 * \brief Thread utility for reading memory using cub::CacheLoadModifier cache modifiers.  Can be used to load any data type.
 *
 * \par Example
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/thread/thread_load.cuh>
 *
 * // 32-bit load using cache-global modifier:
 * int *d_in;
 * int val = cub::ThreadLoad<cub::LOAD_CA>(d_in + threadIdx.x);
 *
 * // 16-bit load using default modifier
 * short *d_in;
 * short val = cub::ThreadLoad<cub::LOAD_DEFAULT>(d_in + threadIdx.x);
 *
 * // 256-bit load using cache-volatile modifier
 * double4 *d_in;
 * double4 val = cub::ThreadLoad<cub::LOAD_CV>(d_in + threadIdx.x);
 *
 * // 96-bit load using cache-streaming modifier
 * struct TestFoo { bool a; short b; };
 * TestFoo *d_struct;
 * TestFoo val = cub::ThreadLoad<cub::LOAD_CS>(d_in + threadIdx.x);
 * \endcode
 *
 * \tparam MODIFIER             <b>[inferred]</b> CacheLoadModifier enumeration
 * \tparam InputIteratorT       <b>[inferred]</b> Input iterator type \iterator
 */
template <
    CacheLoadModifier MODIFIER,
    typename InputIteratorT>
__device__ __forceinline__ typename std::iterator_traits<InputIteratorT>::value_type ThreadLoad(InputIteratorT itr);


//@}  end member group


#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


/// Helper structure for templated load iteration (inductive case)
template <int COUNT, int MAX>
struct IterateThreadLoad
{
    template <CacheLoadModifier MODIFIER, typename T>
    static __device__ __forceinline__ void Load(T *ptr, T *vals)
    {
        vals[COUNT] = ThreadLoad<MODIFIER>(ptr + COUNT);
        IterateThreadLoad<COUNT + 1, MAX>::template Load<MODIFIER>(ptr, vals);
    }

    template <typename InputIteratorT, typename T>
    static __device__ __forceinline__ void Dereference(InputIteratorT ptr, T *vals)
    {
        vals[COUNT] = ptr[COUNT];
        IterateThreadLoad<COUNT + 1, MAX>::Dereference(ptr, vals);
    }
};


/// Helper structure for templated load iteration (termination case)
template <int MAX>
struct IterateThreadLoad<MAX, MAX>
{
    template <CacheLoadModifier MODIFIER, typename T>
    static __device__ __forceinline__ void Load(T *ptr, T *vals) {}

    template <typename InputIteratorT, typename T>
    static __device__ __forceinline__ void Dereference(InputIteratorT ptr, T *vals) {}
};


/**
 * Define a uint4 (16B) ThreadLoad specialization for the given Cache load modifier
 */
#define CUB_LOAD_16(cub_modifier, ptx_modifier)                                             \
    template<>                                                                              \
    __device__ __forceinline__ uint4 ThreadLoad<cub_modifier, uint4*>(uint4* ptr)           \
    {                                                                                       \
        uint4 retval;                                                                       \
        asm volatile ("ld."#ptx_modifier".v4.u32 {%0, %1, %2, %3}, [%4];" :                 \
            "=r"(retval.x),                                                                 \
            "=r"(retval.y),                                                                 \
            "=r"(retval.z),                                                                 \
            "=r"(retval.w) :                                                                \
            _CUB_ASM_PTR_(ptr));                                                            \
        return retval;                                                                      \
    }                                                                                       \
    template<>                                                                              \
    __device__ __forceinline__ ulonglong2 ThreadLoad<cub_modifier, ulonglong2*>(ulonglong2* ptr)              \
    {                                                                                       \
        ulonglong2 retval;                                                                  \
        asm volatile ("ld."#ptx_modifier".v2.u64 {%0, %1}, [%2];" :                         \
            "=l"(retval.x),                                                                 \
            "=l"(retval.y) :                                                                \
            _CUB_ASM_PTR_(ptr));                                                            \
        return retval;                                                                      \
    }

/**
 * Define a uint2 (8B) ThreadLoad specialization for the given Cache load modifier
 */
#define CUB_LOAD_8(cub_modifier, ptx_modifier)                                              \
    template<>                                                                              \
    __device__ __forceinline__ ushort4 ThreadLoad<cub_modifier, ushort4*>(ushort4* ptr)     \
    {                                                                                       \
        ushort4 retval;                                                                     \
        asm volatile ("ld."#ptx_modifier".v4.u16 {%0, %1, %2, %3}, [%4];" :                 \
            "=h"(retval.x),                                                                 \
            "=h"(retval.y),                                                                 \
            "=h"(retval.z),                                                                 \
            "=h"(retval.w) :                                                                \
            _CUB_ASM_PTR_(ptr));                                                            \
        return retval;                                                                      \
    }                                                                                       \
    template<>                                                                              \
    __device__ __forceinline__ uint2 ThreadLoad<cub_modifier, uint2*>(uint2* ptr)           \
    {                                                                                       \
        uint2 retval;                                                                       \
        asm volatile ("ld."#ptx_modifier".v2.u32 {%0, %1}, [%2];" :                         \
            "=r"(retval.x),                                                                 \
            "=r"(retval.y) :                                                                \
            _CUB_ASM_PTR_(ptr));                                                            \
        return retval;                                                                      \
    }                                                                                       \
    template<>                                                                              \
    __device__ __forceinline__ unsigned long long ThreadLoad<cub_modifier, unsigned long long*>(unsigned long long* ptr)                 \
    {                                                                                       \
        unsigned long long retval;                                                          \
        asm volatile ("ld."#ptx_modifier".u64 %0, [%1];" :                                  \
            "=l"(retval) :                                                                  \
            _CUB_ASM_PTR_(ptr));                                                            \
        return retval;                                                                      \
    }

/**
 * Define a uint (4B) ThreadLoad specialization for the given Cache load modifier
 */
#define CUB_LOAD_4(cub_modifier, ptx_modifier)                                              \
    template<>                                                                              \
    __device__ __forceinline__ unsigned int ThreadLoad<cub_modifier, unsigned int*>(unsigned int* ptr)                 \
    {                                                                                       \
        unsigned int retval;                                                                \
        asm volatile ("ld."#ptx_modifier".u32 %0, [%1];" :                                  \
            "=r"(retval) :                                                                  \
            _CUB_ASM_PTR_(ptr));                                                            \
        return retval;                                                                      \
    }


/**
 * Define a unsigned short (2B) ThreadLoad specialization for the given Cache load modifier
 */
#define CUB_LOAD_2(cub_modifier, ptx_modifier)                                              \
    template<>                                                                              \
    __device__ __forceinline__ unsigned short ThreadLoad<cub_modifier, unsigned short*>(unsigned short* ptr)           \
    {                                                                                       \
        unsigned short retval;                                                              \
        asm volatile ("ld."#ptx_modifier".u16 %0, [%1];" :                                  \
            "=h"(retval) :                                                                  \
            _CUB_ASM_PTR_(ptr));                                                            \
        return retval;                                                                      \
    }


/**
 * Define an unsigned char (1B) ThreadLoad specialization for the given Cache load modifier
 */
#define CUB_LOAD_1(cub_modifier, ptx_modifier)                                              \
    template<>                                                                              \
    __device__ __forceinline__ unsigned char ThreadLoad<cub_modifier, unsigned char*>(unsigned char* ptr)              \
    {                                                                                       \
        unsigned short retval;                                                              \
        asm volatile (                                                                      \
        "{"                                                                                 \
        "   .reg .u8 datum;"                                                                \
        "    ld."#ptx_modifier".u8 datum, [%1];"                                            \
        "    cvt.u16.u8 %0, datum;"                                                         \
        "}" :                                                                               \
            "=h"(retval) :                                                                  \
            _CUB_ASM_PTR_(ptr));                                                            \
        return (unsigned char) retval;                                                               \
    }


/**
 * Define powers-of-two ThreadLoad specializations for the given Cache load modifier
 */
#define CUB_LOAD_ALL(cub_modifier, ptx_modifier)                                            \
    CUB_LOAD_16(cub_modifier, ptx_modifier)                                                 \
    CUB_LOAD_8(cub_modifier, ptx_modifier)                                                  \
    CUB_LOAD_4(cub_modifier, ptx_modifier)                                                  \
    CUB_LOAD_2(cub_modifier, ptx_modifier)                                                  \
    CUB_LOAD_1(cub_modifier, ptx_modifier)                                                  \


/**
 * Define powers-of-two ThreadLoad specializations for the various Cache load modifiers
 */
#if CUB_PTX_ARCH >= 200
    CUB_LOAD_ALL(LOAD_CA, ca)
    CUB_LOAD_ALL(LOAD_CG, cg)
    CUB_LOAD_ALL(LOAD_CS, cs)
    CUB_LOAD_ALL(LOAD_CV, cv)
#else
    CUB_LOAD_ALL(LOAD_CA, global)
    // Use volatile to ensure coherent reads when this PTX is JIT'd to run on newer architectures with L1
    CUB_LOAD_ALL(LOAD_CG, volatile.global)
    CUB_LOAD_ALL(LOAD_CS, global)
    CUB_LOAD_ALL(LOAD_CV, volatile.global)
#endif

#if CUB_PTX_ARCH >= 350
    CUB_LOAD_ALL(LOAD_LDG, global.nc)
#else
    CUB_LOAD_ALL(LOAD_LDG, global)
#endif


// Macro cleanup
#undef CUB_LOAD_ALL
#undef CUB_LOAD_1
#undef CUB_LOAD_2
#undef CUB_LOAD_4
#undef CUB_LOAD_8
#undef CUB_LOAD_16



/**
 * ThreadLoad definition for LOAD_DEFAULT modifier on iterator types
 */
template <typename InputIteratorT>
__device__ __forceinline__ typename std::iterator_traits<InputIteratorT>::value_type ThreadLoad(
    InputIteratorT          itr,
    Int2Type<LOAD_DEFAULT>  modifier,
    Int2Type<false>         is_pointer)
{
    return *itr;
}


/**
 * ThreadLoad definition for LOAD_DEFAULT modifier on pointer types
 */
template <typename T>
__device__ __forceinline__ T ThreadLoad(
    T                       *ptr,
    Int2Type<LOAD_DEFAULT>  modifier,
    Int2Type<true>          is_pointer)
{
    return *ptr;
}


/**
 * ThreadLoad definition for LOAD_VOLATILE modifier on primitive pointer types
 */
template <typename T>
__device__ __forceinline__ T ThreadLoadVolatilePointer(
    T                       *ptr,
    Int2Type<true>          is_primitive)
{
    T retval = *reinterpret_cast<volatile T*>(ptr);

#if (CUB_PTX_ARCH <= 130)
    if (sizeof(T) == 1) __threadfence_block();
#endif

    return retval;
}


/**
 * ThreadLoad definition for LOAD_VOLATILE modifier on non-primitive pointer types
 */
template <typename T>
__device__ __forceinline__ T ThreadLoadVolatilePointer(
    T                       *ptr,
    Int2Type<false>          is_primitive)
{

#if CUB_PTX_ARCH <= 130

    T retval = *ptr;
    __threadfence_block();
    return retval;

#else

    typedef typename UnitWord<T>::VolatileWord VolatileWord;   // Word type for memcopying

    const int VOLATILE_MULTIPLE = sizeof(T) / sizeof(VolatileWord);
/*
    VolatileWord words[VOLATILE_MULTIPLE];

    IterateThreadLoad<0, VOLATILE_MULTIPLE>::Dereference(
        reinterpret_cast<volatile VolatileWord*>(ptr),
        words);

    return *reinterpret_cast<T*>(words);
*/

    T retval;
    VolatileWord *words = reinterpret_cast<VolatileWord*>(&retval);
    IterateThreadLoad<0, VOLATILE_MULTIPLE>::Dereference(
        reinterpret_cast<volatile VolatileWord*>(ptr),
        words);
    return retval;

#endif  // CUB_PTX_ARCH <= 130
}


/**
 * ThreadLoad definition for LOAD_VOLATILE modifier on pointer types
 */
template <typename T>
__device__ __forceinline__ T ThreadLoad(
    T                       *ptr,
    Int2Type<LOAD_VOLATILE> modifier,
    Int2Type<true>          is_pointer)
{
    // Apply tags for partial-specialization
    return ThreadLoadVolatilePointer(ptr, Int2Type<Traits<T>::PRIMITIVE>());
}


/**
 * ThreadLoad definition for generic modifiers on pointer types
 */
template <typename T, int MODIFIER>
__device__ __forceinline__ T ThreadLoad(
    T                       *ptr,
    Int2Type<MODIFIER>      modifier,
    Int2Type<true>          is_pointer)
{
    typedef typename UnitWord<T>::DeviceWord DeviceWord;

    const int DEVICE_MULTIPLE = sizeof(T) / sizeof(DeviceWord);

    DeviceWord words[DEVICE_MULTIPLE];

    IterateThreadLoad<0, DEVICE_MULTIPLE>::template Load<CacheLoadModifier(MODIFIER)>(
        reinterpret_cast<DeviceWord*>(ptr),
        words);

    return *reinterpret_cast<T*>(words);
}


/**
 * ThreadLoad definition for generic modifiers
 */
template <
    CacheLoadModifier MODIFIER,
    typename InputIteratorT>
__device__ __forceinline__ typename std::iterator_traits<InputIteratorT>::value_type ThreadLoad(InputIteratorT itr)
{
    // Apply tags for partial-specialization
    return ThreadLoad(
        itr,
        Int2Type<MODIFIER>(),
        Int2Type<IsPointer<InputIteratorT>::VALUE>());
}



#endif // DOXYGEN_SHOULD_SKIP_THIS


/** @} */       // end group UtilIo


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
