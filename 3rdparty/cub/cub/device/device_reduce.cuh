
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
 * cub::DeviceReduce provides device-wide, parallel operations for computing a reduction across a sequence of data items residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "dispatch/dispatch_reduce.cuh"
#include "dispatch/dispatch_reduce_by_key.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \brief DeviceReduce provides device-wide, parallel operations for computing a reduction across a sequence of data items residing within global memory. ![](reduce_logo.png)
 * \ingroup DeviceModule
 *
 * \par Overview
 * A <a href="http://en.wikipedia.org/wiki/Reduce_(higher-order_function)"><em>reduction</em></a> (or <em>fold</em>)
 * uses a binary combining operator to compute a single aggregate from a sequence of input elements.
 *
 * \par Usage Considerations
 * \cdp_class{DeviceReduce}
 *
 * \par Performance
 * \linear_performance{reduction, reduce-by-key, and run-length encode}
 *
 * \par
 * The following chart illustrates DeviceReduce::Sum
 * performance across different CUDA architectures for \p int32 keys.
 *
 * \image html reduce_int32.png
 *
 * \par
 * The following chart illustrates DeviceReduce::ReduceByKey (summation)
 * performance across different CUDA architectures for \p fp32
 * values.  Segments are identified by \p int32 keys, and have lengths uniformly sampled from [1,1000].
 *
 * \image html reduce_by_key_fp32_len_500.png
 *
 * \par
 * \plots_below
 *
 */
struct DeviceReduce
{
    /**
     * \brief Computes a device-wide reduction using the specified binary \p reduction_op functor.
     *
     * \par
     * - Does not support non-commutative reduction operators.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * Performance is typically similar to DeviceReduce::Sum.
     *
     * \par Snippet
     * The code snippet below illustrates a custom min reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // CustomMin functor
     * struct CustomMin
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int          num_items;  // e.g., 7
     * int          *d_in;      // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int          *d_out;     // e.g., [ ]
     * CustomMin    min_op;
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, min_op);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run reduction
     * cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, min_op);
     *
     * // d_out <-- [0]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading input items \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Output iterator type for recording the reduced aggregate \iterator
     * \tparam ReductionOp        <b>[inferred]</b> Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt> (e.g., cub::Sum, cub::Min, cub::Max, etc.)
     */
    template <
        typename                    InputIteratorT,
        typename                    OutputIteratorT,
        typename                    ReductionOp>
    CUB_RUNTIME_FUNCTION
    static cudaError_t Reduce(
        void*                       d_temp_storage,                     ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT              d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT             d_out,                              ///< [out] Pointer to the output aggregate
        int                         num_items,                          ///< [in] Total number of input items (i.e., length of \p d_in)
        ReductionOp                 reduction_op,                       ///< [in] Binary reduction functor (e.g., an instance of cub::Sum, cub::Min, cub::Max, etc.)
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        // Dispatch type
        typedef DispatchReduce<InputIteratorT, OutputIteratorT, OffsetT, ReductionOp> DispatchReduce;

        return DispatchReduce::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_items,
            reduction_op,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide sum using the addition ('+') operator.
     *
     * \par
     * - Does not support non-commutative reduction operators.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * The following charts illustrate saturated reduction (sum) performance across different
     * CUDA architectures for \p int32 and \p int64 items, respectively.
     *
     * \image html reduce_int32.png
     * \image html reduce_int64.png
     *
     * \par Snippet
     * The code snippet below illustrates the sum reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run sum-reduction
     * cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items);
     *
     * // d_out <-- [38]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading input items \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Output iterator type for recording the reduced aggregate \iterator
     */
    template <
        typename                    InputIteratorT,
        typename                    OutputIteratorT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t Sum(
        void*                       d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT              d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT             d_out,                              ///< [out] Pointer to the output aggregate
        int                         num_items,                          ///< [in] Total number of input items (i.e., length of \p d_in)
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        // Dispatch type
        typedef DispatchReduce<InputIteratorT, OutputIteratorT, OffsetT, cub::Sum> DispatchReduce;

        return DispatchReduce::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_items,
            cub::Sum(),
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide minimum using the less-than ('<') operator.
     *
     * \par
     * - Does not support non-commutative minimum operators.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * Performance is typically similar to DeviceReduce::Sum.
     *
     * \par Snippet
     * The code snippet below illustrates the min-reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_min, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run min-reduction
     * cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_min, num_items);
     *
     * // d_out <-- [0]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading input items \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Output iterator type for recording the reduced aggregate \iterator
     */
    template <
        typename                    InputIteratorT,
        typename                    OutputIteratorT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t Min(
        void*                       d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT              d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT             d_out,                              ///< [out] Pointer to the output aggregate
        int                         num_items,                          ///< [in] Total number of input items (i.e., length of \p d_in)
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        // Dispatch type
        typedef DispatchReduce<InputIteratorT, OutputIteratorT, OffsetT, cub::Min> DispatchReduce;

        return DispatchReduce::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_items,
            cub::Min(),
            stream,
            debug_synchronous);
    }


    /**
     * \brief Finds the first device-wide minimum using the less-than ('<') operator, also returning the index of that item.
     *
     * \par
     * Assuming the input \p d_in has value type \p T, the output \p d_out must have value type
     * <tt>KeyValuePair<int, T></tt>.  The minimum value is written to <tt>d_out.value</tt> and its
     * location in the input array is written to <tt>d_out.key</tt>.
     *
     * \par
     * - Does not support non-commutative minimum operators.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * Performance is typically similar to DeviceReduce::Sum.
     *
     * \par Snippet
     * The code snippet below illustrates the argmin-reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int                      num_items;      // e.g., 7
     * int                      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * KeyValuePair<int, int>   *d_out;         // e.g., [{ , }]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_argmin, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run argmin-reduction
     * cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_argmin, num_items);
     *
     * // d_out <-- [{0, 5}]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading input items (of some type \p T) \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Output iterator type for recording the reduced aggregate (having value type <tt>KeyValuePair<int, T></tt>) \iterator
     */
    template <
        typename                    InputIteratorT,
        typename                    OutputIteratorT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t ArgMin(
        void*               d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT              d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT             d_out,                              ///< [out] Pointer to the output aggregate
        int                         num_items,                          ///< [in] Total number of input items (i.e., length of \p d_in)
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        // Wrapped input iterator
        typedef ArgIndexInputIterator<InputIteratorT, int> ArgIndexInputIteratorT;
        ArgIndexInputIteratorT d_argmin_in(d_in, 0);

        // Dispatch type
        typedef DispatchReduce<ArgIndexInputIteratorT, OutputIteratorT, OffsetT, cub::ArgMin> DispatchReduce;

        return DispatchReduce::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_argmin_in,
            d_out,
            num_items,
            cub::ArgMin(),
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide maximum using the greater-than ('>') operator.
     *
     * \par
     * - Does not support non-commutative maximum operators.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * Performance is typically similar to DeviceReduce::Sum.
     *
     * \par Snippet
     * The code snippet below illustrates the max-reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run max-reduction
     * cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);
     *
     * // d_out <-- [9]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading input items \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Output iterator type for recording the reduced aggregate \iterator
     */
    template <
        typename                    InputIteratorT,
        typename                    OutputIteratorT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t Max(
        void*               d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT              d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT             d_out,                              ///< [out] Pointer to the output aggregate
        int                         num_items,                          ///< [in] Total number of input items (i.e., length of \p d_in)
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        // Dispatch type
        typedef DispatchReduce<InputIteratorT, OutputIteratorT, OffsetT, cub::Max> DispatchReduce;

        return DispatchReduce::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            num_items,
            cub::Max(),
            stream,
            debug_synchronous);
    }


    /**
     * \brief Finds the first device-wide maximum using the greater-than ('>') operator, also returning the index of that item
     *
     * \par
     * Assuming the input \p d_in has value type \p T, the output \p d_out must have value type
     * <tt>KeyValuePair<int, T></tt>.  The maximum value is written to <tt>d_out.value</tt> and its
     * location in the input array is written to <tt>d_out.key</tt>.
     *
     * \par
     * - Does not support non-commutative maximum operators.
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * Performance is typically similar to DeviceReduce::Sum.
     *
     * \par Snippet
     * The code snippet below illustrates the argmax-reduction of a device vector of \p int items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_reduce.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int                      num_items;      // e.g., 7
     * int                      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * KeyValuePair<int, int>   *d_out;         // e.g., [{ , }]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_argmax, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run argmax-reduction
     * cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_argmax, num_items);
     *
     * // d_out <-- [{9, 6}]
     *
     * \endcode
     *
     * \tparam InputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading input items (of some type \p T) \iterator
     * \tparam OutputIteratorT    <b>[inferred]</b> Output iterator type for recording the reduced aggregate (having value type <tt>KeyValuePair<int, T></tt>) \iterator
     */
    template <
        typename                    InputIteratorT,
        typename                    OutputIteratorT>
    CUB_RUNTIME_FUNCTION
    static cudaError_t ArgMax(
        void*               d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,                ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        InputIteratorT              d_in,                               ///< [in] Pointer to the input sequence of data items
        OutputIteratorT             d_out,                              ///< [out] Pointer to the output aggregate
        int                         num_items,                          ///< [in] Total number of input items (i.e., length of \p d_in)
        cudaStream_t                stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int OffsetT;

        // Wrapped input iterator
        typedef ArgIndexInputIterator<InputIteratorT, int> ArgIndexInputIteratorT;
        ArgIndexInputIteratorT d_argmax_in(d_in, 0);

        // Dispatch type
        typedef DispatchReduce<ArgIndexInputIteratorT, OutputIteratorT, OffsetT, cub::ArgMax> DispatchReduce;

        return DispatchReduce::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_argmax_in,
            d_out,
            num_items,
            cub::ArgMax(),
            stream,
            debug_synchronous);
    }


    /**
     * \brief Reduces segments of values, where segments are demarcated by corresponding runs of identical keys.
     *
     * \par
     * This operation computes segmented reductions within \p d_values_in using
     * the specified binary \p reduction_op functor.  The segments are identified by
     * "runs" of corresponding keys in \p d_keys_in, where runs are maximal ranges of
     * consecutive, identical keys.  For the <em>i</em><sup>th</sup> run encountered,
     * the first key of the run and the corresponding value aggregate of that run are
     * written to <tt>d_unique_out[<em>i</em>]</tt> and <tt>d_aggregates_out[<em>i</em>]</tt>,
     * respectively. The total number of runs encountered is written to \p d_num_runs_out.
     *
     * \par
     * - The <tt>==</tt> equality operator is used to determine whether keys are equivalent
     * - \devicestorage
     * - \cdp
     *
     * \par Performance
     * The following chart illustrates reduction-by-key (sum) performance across
     * different CUDA architectures for \p fp32 and \p fp64 values, respectively.  Segments
     * are identified by \p int32 keys, and have lengths uniformly sampled from [1,1000].
     *
     * \image html reduce_by_key_fp32_len_500.png
     * \image html reduce_by_key_fp64_len_500.png
     *
     * \par
     * The following charts are similar, but with segment lengths uniformly sampled from [1,10]:
     *
     * \image html reduce_by_key_fp32_len_5.png
     * \image html reduce_by_key_fp64_len_5.png
     *
     * \par Snippet
     * The code snippet below illustrates the segmented reduction of \p int values grouped
     * by runs of associated \p int keys.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_reduce.cuh>
     *
     * // CustomMin functor
     * struct CustomMin
     * {
     *     template <typename T>
     *     CUB_RUNTIME_FUNCTION __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int          num_items;          // e.g., 8
     * int          *d_keys_in;         // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
     * int          *d_values_in;       // e.g., [0, 7, 1, 6, 2, 5, 3, 4]
     * int          *d_unique_out;      // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * int          *d_aggregates_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
     * int          *d_num_runs_out;        // e.g., [ ]
     * CustomMin    reduction_op;
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_unique_out, d_values_in, d_aggregates_out, d_num_runs_out, reduction_op, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run reduce-by-key
     * cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_in, d_unique_out, d_values_in, d_aggregates_out, d_num_runs_out, reduction_op, num_items);
     *
     * // d_unique_out      <-- [0, 2, 9, 5, 8]
     * // d_aggregates_out  <-- [0, 1, 6, 2, 4]
     * // d_num_runs_out        <-- [5]
     *
     * \endcode
     *
     * \tparam KeysInputIteratorT       <b>[inferred]</b> Random-access input iterator type for reading input keys \iterator
     * \tparam UniqueOutputIteratorT    <b>[inferred]</b> Random-access output iterator type for writing unique output keys \iterator
     * \tparam ValuesInputIteratorT     <b>[inferred]</b> Random-access input iterator type for reading input values \iterator
     * \tparam AggregatesOutputIterator <b>[inferred]</b> Random-access output iterator type for writing output value aggregates \iterator
     * \tparam NumRunsOutputIteratorT   <b>[inferred]</b> Output iterator type for recording the number of runs encountered \iterator
     * \tparam ReductionOp              <b>[inferred]</b> Binary reduction functor type having member <tt>T operator()(const T &a, const T &b)</tt> (e.g., cub::Sum, cub::Min, cub::Max, etc.)
     */
    template <
        typename                    KeysInputIteratorT,
        typename                    UniqueOutputIteratorT,
        typename                    ValuesInputIteratorT,
        typename                    AggregatesOutputIteratorT,
        typename                    NumRunsOutputIteratorT,
        typename                    ReductionOp>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t ReduceByKey(
        void*               d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,            ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        KeysInputIteratorT          d_keys_in,                      ///< [in] Pointer to the input sequence of keys
        UniqueOutputIteratorT       d_unique_out,                   ///< [out] Pointer to the output sequence of unique keys (one key per run)
        ValuesInputIteratorT        d_values_in,                    ///< [in] Pointer to the input sequence of corresponding values
        AggregatesOutputIteratorT   d_aggregates_out,               ///< [out] Pointer to the output sequence of value aggregates (one aggregate per run)
        NumRunsOutputIteratorT      d_num_runs_out,                     ///< [out] Pointer to total number of runs encountered (i.e., the length of d_unique_out)
        ReductionOp                 reduction_op,                   ///< [in] Binary reduction functor (e.g., an instance of cub::Sum, cub::Min, cub::Max, etc.)
        int                         num_items,                      ///< [in] Total number of associated key+value pairs (i.e., the length of \p d_in_keys and \p d_in_values)
        cudaStream_t                stream             = 0,         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous  = false)     ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        typedef int                 OffsetT;         // Signed integer type for global offsets
        typedef NullType*           FlagIterator;   // FlagT iterator type (not used)
        typedef NullType            SelectOp;       // Selection op (not used)
        typedef Equality            EqualityOp;     // Default == operator

        return DispatchReduceByKey<KeysInputIteratorT, UniqueOutputIteratorT, ValuesInputIteratorT, AggregatesOutputIteratorT, NumRunsOutputIteratorT, EqualityOp, ReductionOp, OffsetT>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_keys_in,
            d_unique_out,
            d_values_in,
            d_aggregates_out,
            d_num_runs_out,
            EqualityOp(),
            reduction_op,
            num_items,
            stream,
            debug_synchronous);
    }

};

/**
 * \example example_device_reduce.cu
 */

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


