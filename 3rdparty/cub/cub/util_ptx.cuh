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
 * PTX intrinsics
 */


#pragma once

#include "util_type.cuh"
#include "util_arch.cuh"
#include "util_namespace.cuh"
#include "util_debug.cuh"


/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup UtilPtx
 * @{
 */


/******************************************************************************
 * PTX helper macros
 ******************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Register modifier for pointer-types (for inlining PTX assembly)
 */
#if defined(_WIN64) || defined(__LP64__)
    #define __CUB_LP64__ 1
    // 64-bit register modifier for inlined asm
    #define _CUB_ASM_PTR_ "l"
    #define _CUB_ASM_PTR_SIZE_ "u64"
#else
    #define __CUB_LP64__ 0
    // 32-bit register modifier for inlined asm
    #define _CUB_ASM_PTR_ "r"
    #define _CUB_ASM_PTR_SIZE_ "u32"
#endif

#endif // DOXYGEN_SHOULD_SKIP_THIS


/******************************************************************************
 * Inlined PTX intrinsics
 ******************************************************************************/

/**
 * \brief Shift-right then add.  Returns (\p x >> \p shift) + \p addend.
 */
__device__ __forceinline__ unsigned int SHR_ADD(
    unsigned int x,
    unsigned int shift,
    unsigned int addend)
{
    unsigned int ret;
#if CUB_PTX_ARCH >= 200
    asm volatile("vshr.u32.u32.u32.clamp.add %0, %1, %2, %3;" :
        "=r"(ret) : "r"(x), "r"(shift), "r"(addend));
#else
    ret = (x >> shift) + addend;
#endif
    return ret;
}


/**
 * \brief Shift-left then add.  Returns (\p x << \p shift) + \p addend.
 */
__device__ __forceinline__ unsigned int SHL_ADD(
    unsigned int x,
    unsigned int shift,
    unsigned int addend)
{
    unsigned int ret;
#if CUB_PTX_ARCH >= 200
    asm volatile("vshl.u32.u32.u32.clamp.add %0, %1, %2, %3;" :
        "=r"(ret) : "r"(x), "r"(shift), "r"(addend));
#else
    ret = (x << shift) + addend;
#endif
    return ret;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Bitfield-extract.
 */
template <typename UnsignedBits, int BYTE_LEN>
__device__ __forceinline__ unsigned int BFE(
    UnsignedBits            source,
    unsigned int            bit_start,
    unsigned int            num_bits,
    Int2Type<BYTE_LEN>      byte_len)
{
    unsigned int bits;
#if CUB_PTX_ARCH >= 200
    asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(bit_start), "r"(num_bits));
#else
    const unsigned int MASK = (1 << num_bits) - 1;
    bits = (source >> bit_start) & MASK;
#endif
    return bits;
}


/**
 * Bitfield-extract for 64-bit types.
 */
template <typename UnsignedBits>
__device__ __forceinline__ unsigned int BFE(
    UnsignedBits            source,
    unsigned int            bit_start,
    unsigned int            num_bits,
    Int2Type<8>             byte_len)
{
    const unsigned long long MASK = (1ull << num_bits) - 1;
    return (source >> bit_start) & MASK;
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Bitfield-extract.  Extracts \p num_bits from \p source starting at bit-offset \p bit_start.  The input \p source may be an 8b, 16b, 32b, or 64b unsigned integer type.
 */
template <typename UnsignedBits>
__device__ __forceinline__ unsigned int BFE(
    UnsignedBits source,
    unsigned int bit_start,
    unsigned int num_bits)
{
    return BFE(source, bit_start, num_bits, Int2Type<sizeof(UnsignedBits)>());
}


/**
 * \brief Bitfield insert.  Inserts the \p num_bits least significant bits of \p y into \p x at bit-offset \p bit_start.
 */
__device__ __forceinline__ void BFI(
    unsigned int &ret,
    unsigned int x,
    unsigned int y,
    unsigned int bit_start,
    unsigned int num_bits)
{
#if CUB_PTX_ARCH >= 200
    asm volatile("bfi.b32 %0, %1, %2, %3, %4;" :
        "=r"(ret) : "r"(y), "r"(x), "r"(bit_start), "r"(num_bits));
#else
    x <<= bit_start;
    unsigned int MASK_X = ((1 << num_bits) - 1) << bit_start;
    unsigned int MASK_Y = ~MASK_X;
    ret = (y & MASK_Y) | (x & MASK_X);
#endif
}


/**
 * \brief Three-operand add.  Returns \p x + \p y + \p z.
 */
__device__ __forceinline__ unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z)
{
#if CUB_PTX_ARCH >= 200
    asm volatile("vadd.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(x) : "r"(x), "r"(y), "r"(z));
#else
    x = x + y + z;
#endif
    return x;
}


/**
 * \brief Byte-permute. Pick four arbitrary bytes from two 32-bit registers, and reassemble them into a 32-bit destination register.  For SM2.0 or later.
 *
 * \par
 * The bytes in the two source registers \p a and \p b are numbered from 0 to 7:
 * {\p b, \p a} = {{b7, b6, b5, b4}, {b3, b2, b1, b0}}. For each of the four bytes
 * {b3, b2, b1, b0} selected in the return value, a 4-bit selector is defined within
 * the four lower "nibbles" of \p index: {\p index } = {n7, n6, n5, n4, n3, n2, n1, n0}
 *
 * \par Snippet
 * The code snippet below illustrates byte-permute.
 * \par
 * \code
 * #include <cub/cub.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     int a        = 0x03020100;
 *     int b        = 0x07060504;
 *     int index    = 0x00007531;
 *
 *     int selected = PRMT(a, b, index);    // 0x07050301
 *
 * \endcode
 *
 */
__device__ __forceinline__ int PRMT(unsigned int a, unsigned int b, unsigned int index)
{
    int ret;
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(a), "r"(b), "r"(index));
    return ret;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Sync-threads barrier.
 */
__device__ __forceinline__ void BAR(int count)
{
    asm volatile("bar.sync 1, %0;" : : "r"(count));
}


/**
 * Floating point multiply. (Mantissa LSB rounds towards zero.)
 */
__device__ __forceinline__ float FMUL_RZ(float a, float b)
{
    float d;
    asm volatile("mul.rz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}


/**
 * Floating point multiply-add. (Mantissa LSB rounds towards zero.)
 */
__device__ __forceinline__ float FFMA_RZ(float a, float b, float c)
{
    float d;
    asm volatile("fma.rz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Terminates the calling thread
 */
__device__ __forceinline__ void ThreadExit() {
    asm volatile("exit;");
}    


/**
 * \brief Returns the row-major linear thread identifier for a multidimensional threadblock
 */
__device__ __forceinline__ int RowMajorTid(int block_dim_x, int block_dim_y, int block_dim_z)
{
    return ((block_dim_z == 1) ? 0 : (threadIdx.z * block_dim_x * block_dim_y)) +
            ((block_dim_y == 1) ? 0 : (threadIdx.y * block_dim_x)) +
            threadIdx.x;
}


/**
 * \brief Returns the warp lane ID of the calling thread
 */
__device__ __forceinline__ unsigned int LaneId()
{
    unsigned int ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret) );
    return ret;
}


/**
 * \brief Returns the warp ID of the calling thread.  Warp ID is guaranteed to be unique among warps, but may not correspond to a zero-based ranking within the thread block.
 */
__device__ __forceinline__ unsigned int WarpId()
{
    unsigned int ret;
    asm volatile("mov.u32 %0, %warpid;" : "=r"(ret) );
    return ret;
}

/**
 * \brief Returns the warp lane mask of all lanes less than the calling thread
 */
__device__ __forceinline__ unsigned int LaneMaskLt()
{
    unsigned int ret;
    asm volatile("mov.u32 %0, %lanemask_lt;" : "=r"(ret) );
    return ret;
}

/**
 * \brief Returns the warp lane mask of all lanes less than or equal to the calling thread
 */
__device__ __forceinline__ unsigned int LaneMaskLe()
{
    unsigned int ret;
    asm volatile("mov.u32 %0, %lanemask_le;" : "=r"(ret) );
    return ret;
}

/**
 * \brief Returns the warp lane mask of all lanes greater than the calling thread
 */
__device__ __forceinline__ unsigned int LaneMaskGt()
{
    unsigned int ret;
    asm volatile("mov.u32 %0, %lanemask_gt;" : "=r"(ret) );
    return ret;
}

/**
 * \brief Returns the warp lane mask of all lanes greater than or equal to the calling thread
 */
__device__ __forceinline__ unsigned int LaneMaskGe()
{
    unsigned int ret;
    asm volatile("mov.u32 %0, %lanemask_ge;" : "=r"(ret) );
    return ret;
}

/** @} */       // end group UtilPtx



#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


/**
 * Shuffle word up
 */
template <typename ShuffleWordT, int STEP>
__device__ __forceinline__ void ShuffleUp(
    ShuffleWordT*   input, 
    ShuffleWordT*   output,
    int             src_offset,
    int             first_lane,
    Int2Type<STEP>  step)
{
    unsigned int word = input[STEP];
    asm volatile("shfl.up.b32 %0, %1, %2, %3;"
        : "=r"(word) : "r"(word), "r"(src_offset), "r"(first_lane));
    output[STEP] = (ShuffleWordT) word;

    ShuffleUp(input, output, src_offset, first_lane, Int2Type<STEP - 1>());
}


/**
 * Shuffle word up
 */
template <typename ShuffleWordT>
__device__ __forceinline__ void ShuffleUp(
    ShuffleWordT*   input, 
    ShuffleWordT*   output,
    int             src_offset,
    int             first_lane,
    Int2Type<-1>    step)
{}



/**
 * Shuffle word down
 */
template <typename ShuffleWordT, int STEP>
__device__ __forceinline__ void ShuffleDown(
    ShuffleWordT*   input, 
    ShuffleWordT*   output,
    int             src_offset,
    int             last_lane,
    Int2Type<STEP>  step)
{
    unsigned int word = input[STEP];
    asm volatile("shfl.down.b32 %0, %1, %2, %3;"
        : "=r"(word) : "r"(word), "r"(src_offset), "r"(last_lane));
    output[STEP] = (ShuffleWordT) word;

    ShuffleDown(input, output, src_offset, last_lane, Int2Type<STEP - 1>());
}


/**
 * Shuffle word down
 */
template <typename ShuffleWordT>
__device__ __forceinline__ void ShuffleDown(
    ShuffleWordT*   input, 
    ShuffleWordT*   output,
    int             src_offset,
    int             last_lane,
    Int2Type<-1>    step)
{}


/**
 * Shuffle index
 */
template <typename ShuffleWordT, int STEP>
__device__ __forceinline__ void ShuffleIdx(
    ShuffleWordT*   input, 
    ShuffleWordT*   output,
    int             src_lane,
    int             last_lane,
    Int2Type<STEP>  step)
{
    unsigned int word = input[STEP];
    asm volatile("shfl.idx.b32 %0, %1, %2, %3;"
        : "=r"(word) : "r"(word), "r"(src_lane), "r"(last_lane));
    output[STEP] = (ShuffleWordT) word;

    ShuffleIdx(input, output, src_lane, last_lane, Int2Type<STEP - 1>());
}


/**
 * Shuffle index
 */
template <typename ShuffleWordT>
__device__ __forceinline__ void ShuffleIdx(
    ShuffleWordT*   input, 
    ShuffleWordT*   output,
    int             src_lane,
    int             last_lane,
    Int2Type<-1>    step)
{}




#endif  // DOXYGEN_SHOULD_SKIP_THIS    // Do not document



/**
 * \brief Shuffle-up for any data type.  Each <em>warp-lane<sub>i</sub></em> obtains the value \p input contributed by <em>warp-lane</em><sub><em>i</em>-<tt>src_offset</tt></sub>.  For thread lanes \e i < src_offset, the thread's own \p input is returned to the thread. ![](shfl_up_logo.png)
 * \ingroup WarpModule
 *
 * \par
 * - Available only for SM3.0 or newer
 *
 * \par Snippet
 * The code snippet below illustrates each thread obtaining a \p double value from the
 * predecessor of its predecessor.
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_ptx.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Obtain one input item per thread
 *     double thread_data = ...
 *
 *     // Obtain item from two ranks below
 *     double peer_data = ShuffleUp(thread_data, 2);
 *
 * \endcode
 * \par
 * Suppose the set of input \p thread_data across the first warp of threads is <tt>{1.0, 2.0, 3.0, 4.0, 5.0, ..., 32.0}</tt>.
 * The corresponding output \p peer_data will be <tt>{1.0, 2.0, 1.0, 2.0, 3.0, ..., 30.0}</tt>.
 *
 */
template <typename T>
__device__ __forceinline__ T ShuffleUp(
    T               input,              ///< [in] The value to broadcast
    int             src_offset,         ///< [in] The relative down-offset of the peer to read from
    int             first_lane = 0)     ///< [in] Index of first lane in segment
{
    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);
 
    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    unsigned int shuffle_word;
    asm volatile("shfl.up.b32 %0, %1, %2, %3;"
        : "=r"(shuffle_word) : "r"((unsigned int) input_alias[0]), "r"(src_offset), "r"(first_lane));
    output_alias[0] = shuffle_word;

    #pragma unroll
    for (int WORD = 1; WORD < WORDS; ++WORD)
    {
        asm volatile("shfl.up.b32 %0, %1, %2, %3;"
            : "=r"(shuffle_word) : "r"((unsigned int) input_alias[WORD]), "r"(src_offset), "r"(first_lane));
        output_alias[WORD] = shuffle_word;
    }

//    ShuffleUp(input_alias, output_alias, src_offset, first_lane, Int2Type<WORDS - 1>());

    return output;
}


/**
 * \brief Shuffle-down for any data type.  Each <em>warp-lane<sub>i</sub></em> obtains the value \p input contributed by <em>warp-lane</em><sub><em>i</em>+<tt>src_offset</tt></sub>.  For thread lanes \e i >= WARP_THREADS, the thread's own \p input is returned to the thread.  ![](shfl_down_logo.png)
 * \ingroup WarpModule
 *
 * \par
 * - Available only for SM3.0 or newer
 *
 * \par Snippet
 * The code snippet below illustrates each thread obtaining a \p double value from the
 * successor of its successor.
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_ptx.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Obtain one input item per thread
 *     double thread_data = ...
 *
 *     // Obtain item from two ranks below
 *     double peer_data = ShuffleDown(thread_data, 2);
 *
 * \endcode
 * \par
 * Suppose the set of input \p thread_data across the first warp of threads is <tt>{1.0, 2.0, 3.0, 4.0, 5.0, ..., 32.0}</tt>.
 * The corresponding output \p peer_data will be <tt>{3.0, 4.0, 5.0, 6.0, 7.0, ..., 32.0}</tt>.
 *
 */
template <typename T>
__device__ __forceinline__ T ShuffleDown(
    T               input,                                  ///< [in] The value to broadcast
    int             src_offset,                             ///< [in] The relative up-offset of the peer to read from
    int             last_lane = CUB_PTX_WARP_THREADS - 1)   ///< [in] Index of first lane in segment
{
    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    unsigned int shuffle_word;
    asm volatile("shfl.down.b32 %0, %1, %2, %3;"
        : "=r"(shuffle_word) : "r"((unsigned int) input_alias[0]), "r"(src_offset), "r"(last_lane));
    output_alias[0] = shuffle_word;

    #pragma unroll
    for (int WORD = 1; WORD < WORDS; ++WORD)
    {
        asm volatile("shfl.down.b32 %0, %1, %2, %3;"
            : "=r"(shuffle_word) : "r"((unsigned int) input_alias[WORD]), "r"(src_offset), "r"(last_lane));
        output_alias[WORD] = shuffle_word;
    }

//    ShuffleDown(input_alias, output_alias, src_offset, last_lane, Int2Type<WORDS - 1>());

    return output;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * \brief Shuffle-index for any data type.  Each <em>warp-lane<sub>i</sub></em> obtains the value \p input contributed by <em>warp-lane</em><sub><tt>src_lane</tt></sub>.  For \p src_lane < 0 or \p src_lane >= WARP_THREADS, then the thread's own \p input is returned to the thread.  ![](shfl_broadcast_logo.png)
 * \ingroup WarpModule
 *
 * \par
 * - Available only for SM3.0 or newer
 */
template <typename T>
__device__ __forceinline__ T ShuffleIndex(
    T               input,                                          ///< [in] The value to broadcast
    int             src_lane,                                       ///< [in] Which warp lane is to do the broadcasting
    int             logical_warp_threads)                           ///< [in] Number of threads per logical warp
{
    typedef typename UnitWord<T>::ShuffleWord ShuffleWord;

    const int       WORDS           = (sizeof(T) + sizeof(ShuffleWord) - 1) / sizeof(ShuffleWord);

    T               output;
    ShuffleWord     *output_alias   = reinterpret_cast<ShuffleWord *>(&output);
    ShuffleWord     *input_alias    = reinterpret_cast<ShuffleWord *>(&input);

    unsigned int shuffle_word;
    asm volatile("shfl.idx.b32 %0, %1, %2, %3;"
        : "=r"(shuffle_word) : "r"((unsigned int) input_alias[0]), "r"(src_lane), "r"(logical_warp_threads - 1));
    output_alias[0] = shuffle_word;

    #pragma unroll
    for (int WORD = 1; WORD < WORDS; ++WORD)
    {
        asm volatile("shfl.idx.b32 %0, %1, %2, %3;"
            : "=r"(shuffle_word) : "r"((unsigned int) input_alias[WORD]), "r"(src_lane), "r"(logical_warp_threads - 1));
        output_alias[WORD] = shuffle_word;
    }

//    ShuffleIdx(input_alias, output_alias, src_lane, logical_warp_threads - 1, Int2Type<WORDS - 1>());

    return output;
}

#endif // DOXYGEN_SHOULD_SKIP_THIS


 /**
 * \brief Shuffle-broadcast for any data type.  Each <em>warp-lane<sub>i</sub></em> obtains the value \p input contributed by <em>warp-lane</em><sub><tt>src_lane</tt></sub>.  For \p src_lane < 0 or \p src_lane >= WARP_THREADS, then the thread's own \p input is returned to the thread. ![](shfl_broadcast_logo.png)
 * \ingroup WarpModule
 *
 * \par
 * - Available only for SM3.0 or newer
 *
 * \par Snippet
 * The code snippet below illustrates each thread obtaining a \p double value from <em>warp-lane</em><sub>0</sub>.
 *
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_ptx.cuh>
 *
 * __global__ void ExampleKernel(...)
 * {
 *     // Obtain one input item per thread
 *     double thread_data = ...
 *
 *     // Obtain item from thread 0
 *     double peer_data = ShuffleIndex(thread_data, 0);
 *
 * \endcode
 * \par
 * Suppose the set of input \p thread_data across the first warp of threads is <tt>{1.0, 2.0, 3.0, 4.0, 5.0, ..., 32.0}</tt>.
 * The corresponding output \p peer_data will be <tt>{1.0, 1.0, 1.0, 1.0, 1.0, ..., 1.0}</tt>.
 *
 */
template <typename T>
__device__ __forceinline__ T ShuffleIndex(
    T               input,              ///< [in] The value to broadcast
    int             src_lane)           ///< [in] Which warp lane is to do the broadcasting
{
    return ShuffleIndex(input, src_lane, CUB_PTX_WARP_THREADS);
}





/**
 * \brief Portable implementation of __all
 * \ingroup WarpModule
 */
__device__ __forceinline__ int WarpAll(int cond)
{
#if CUB_PTX_ARCH < 120

    __shared__ volatile int warp_signals[CUB_PTX_MAX_SM_THREADS / CUB_PTX_WARP_THREADS];

    if (LaneId() == 0)
        warp_signals[WarpId()] = 1;

    if (cond == 0)
        warp_signals[WarpId()] = 0;

    return warp_signals[WarpId()];

#else

    return __all(cond);

#endif
}


/**
 * \brief Portable implementation of __any
 * \ingroup WarpModule
 */
__device__ __forceinline__ int WarpAny(int cond)
{
#if CUB_PTX_ARCH < 120

    __shared__ volatile int warp_signals[CUB_PTX_MAX_SM_THREADS / CUB_PTX_WARP_THREADS];

    if (LaneId() == 0)
        warp_signals[WarpId()] = 0;

    if (cond)
        warp_signals[WarpId()] = 1;

    return warp_signals[WarpId()];

#else

    return __any(cond);

#endif
}


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
