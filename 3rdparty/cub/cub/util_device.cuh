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
 * Properties of a given CUDA device and the corresponding PTX bundle
 */

#pragma once

#include "util_arch.cuh"
#include "util_debug.cuh"
#include "util_namespace.cuh"
#include "util_macro.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup UtilMgmt
 * @{
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


/**
 * Empty kernel for querying PTX manifest metadata (e.g., version) for the current device
 */
template <typename T>
__global__ void EmptyKernel(void) { }


/**
 * Alias temporaries to externally-allocated device storage (or simply return the amount of storage needed).
 */
template <int ALLOCATIONS>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t AliasTemporaries(
    void    *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t  &temp_storage_bytes,                ///< [in,out] Size in bytes of \t d_temp_storage allocation
    void*   (&allocations)[ALLOCATIONS],        ///< [in,out] Pointers to device allocations needed
    size_t  (&allocation_sizes)[ALLOCATIONS])   ///< [in] Sizes in bytes of device allocations needed
{
    const int ALIGN_BYTES   = 256;
    const int ALIGN_MASK    = ~(ALIGN_BYTES - 1);

    // Compute exclusive prefix sum over allocation requests
    size_t allocation_offsets[ALLOCATIONS];
    size_t bytes_needed = 0;
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        size_t allocation_bytes = (allocation_sizes[i] + ALIGN_BYTES - 1) & ALIGN_MASK;
        allocation_offsets[i] = bytes_needed;
        bytes_needed += allocation_bytes;
    }

    // Check if the caller is simply requesting the size of the storage allocation
    if (!d_temp_storage)
    {
        temp_storage_bytes = bytes_needed;
        return cudaSuccess;
    }

    // Check if enough storage provided
    if (temp_storage_bytes < bytes_needed)
    {
        return CubDebug(cudaErrorInvalidValue);
    }

    // Alias
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        allocations[i] = static_cast<char*>(d_temp_storage) + allocation_offsets[i];
    }

    return cudaSuccess;
}



#endif  // DOXYGEN_SHOULD_SKIP_THIS



/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10)
 */
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t PtxVersion(int &ptx_version)
{
    struct Dummy
    {
        /// Type definition of the EmptyKernel kernel entry point
        typedef void (*EmptyKernelPtr)();

        /// Force EmptyKernel<void> to be generated if this class is used
        CUB_RUNTIME_FUNCTION __forceinline__
        EmptyKernelPtr Empty()
        {
            return EmptyKernel<void>;
        }
    };


#ifndef CUB_RUNTIME_ENABLED

    // CUDA API calls not supported from this device
    return cudaErrorInvalidConfiguration;

#elif (CUB_PTX_ARCH > 0)

    ptx_version = CUB_PTX_ARCH;
    return cudaSuccess;

#else

    cudaError_t error = cudaSuccess;
    do
    {
        cudaFuncAttributes empty_kernel_attrs;
        if (CubDebug(error = cudaFuncGetAttributes(&empty_kernel_attrs, EmptyKernel<void>))) break;
        ptx_version = empty_kernel_attrs.ptxVersion * 10;
    }
    while (0);

    return error;

#endif
}


/**
 * \brief Retrieves the SM version (major * 100 + minor * 10)
 */
CUB_RUNTIME_FUNCTION __forceinline__ cudaError_t SmVersion(int &sm_version, int device_ordinal)
{
#ifndef CUB_RUNTIME_ENABLED

    // CUDA API calls not supported from this device
    return cudaErrorInvalidConfiguration;

#else

    cudaError_t error = cudaSuccess;
    do
    {
        // Fill in SM version
        int major, minor;
        if (CubDebug(error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_ordinal))) break;
        if (CubDebug(error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_ordinal))) break;
        sm_version = major * 100 + minor * 10;
    }
    while (0);

    return error;

#endif
}


#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/**
 * Synchronize the stream if specified
 */
CUB_RUNTIME_FUNCTION __forceinline__
static cudaError_t SyncStream(cudaStream_t stream)
{
#if (CUB_PTX_ARCH == 0)
    return cudaStreamSynchronize(stream);
#else
    // Device can't yet sync on a specific stream
    return cudaDeviceSynchronize();
#endif
}


/**
 * \brief Computes maximum SM occupancy in thread blocks for the given kernel function pointer \p kernel_ptr.
 */
template <typename KernelPtr>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t MaxSmOccupancy(
    int                 &max_sm_occupancy,          ///< [out] maximum number of thread blocks that can reside on a single SM
    int                 sm_version,                 ///< [in] The SM architecture to run on
    KernelPtr           kernel_ptr,                 ///< [in] Kernel pointer for which to compute SM occupancy
    int                 block_threads)              ///< [in] Number of threads per thread block
{
#ifndef CUB_RUNTIME_ENABLED

    // CUDA API calls not supported from this device
    return CubDebug(cudaErrorInvalidConfiguration);

#else

    return cudaOccupancyMaxActiveBlocksPerMultiprocessor (
        &max_sm_occupancy,
        kernel_ptr,
        block_threads,
        0);
/*
    cudaError_t error = cudaSuccess;
    do
    {
        int warp_threads        = 1 << CUB_LOG_WARP_THREADS(sm_version);
        int max_sm_blocks       = CUB_MAX_SM_BLOCKS(sm_version);
        int max_sm_warps        = CUB_MAX_SM_THREADS(sm_version) / warp_threads;
        int regs_by_block       = CUB_REGS_BY_BLOCK(sm_version);
        int max_sm_registers    = CUB_MAX_SM_REGISTERS(sm_version);
        int warp_alloc_unit     = CUB_WARP_ALLOC_UNIT(sm_version);
        int smem_alloc_unit     = CUB_SMEM_ALLOC_UNIT(sm_version);
        int reg_alloc_unit      = CUB_REG_ALLOC_UNIT(sm_version);
        int smem_bytes          = CUB_SMEM_BYTES(sm_version);

        // Get kernel attributes
        cudaFuncAttributes kernel_attrs;
        if (CubDebug(error = cudaFuncGetAttributes(&kernel_attrs, kernel_ptr))) break;

        // Number of warps per threadblock
        int block_warps = (block_threads +  warp_threads - 1) / warp_threads;

        // Max warp occupancy
        int max_warp_occupancy = (block_warps > 0) ?
            max_sm_warps / block_warps :
            max_sm_blocks;

        // Maximum register occupancy
        int max_reg_occupancy;
        if ((block_threads == 0) || (kernel_attrs.numRegs == 0))
        {
            // Prevent divide-by-zero
            max_reg_occupancy = max_sm_blocks;
        }
        else if (regs_by_block)
        {
            // Allocates registers by threadblock
            int block_regs = CUB_ROUND_UP_NEAREST(kernel_attrs.numRegs * warp_threads * block_warps, reg_alloc_unit);
            max_reg_occupancy = max_sm_registers / block_regs;
        }
        else
        {
            // Allocates registers by warp
            int sm_sides                = warp_alloc_unit;
            int sm_registers_per_side   = max_sm_registers / sm_sides;
            int regs_per_warp           = CUB_ROUND_UP_NEAREST(kernel_attrs.numRegs * warp_threads, reg_alloc_unit);
            int warps_per_side          = sm_registers_per_side / regs_per_warp;
            int warps                   = warps_per_side * sm_sides;
            max_reg_occupancy           = warps / block_warps;
        }

        // Shared memory per threadblock
        int block_allocated_smem = CUB_ROUND_UP_NEAREST(
            (int) kernel_attrs.sharedSizeBytes,
            smem_alloc_unit);

        // Max shared memory occupancy
        int max_smem_occupancy = (block_allocated_smem > 0) ?
            (smem_bytes / block_allocated_smem) :
            max_sm_blocks;

        // Max occupancy
        max_sm_occupancy = CUB_MIN(
            CUB_MIN(max_sm_blocks, max_warp_occupancy),
            CUB_MIN(max_smem_occupancy, max_reg_occupancy));

//            printf("max_smem_occupancy(%d), max_warp_occupancy(%d), max_reg_occupancy(%d) \n", max_smem_occupancy, max_warp_occupancy, max_reg_occupancy);

    } while (0);

    return error;
*/
#endif  // CUB_RUNTIME_ENABLED
}

#endif  // Do not document


/**
 * \brief Computes maximum SM occupancy in thread blocks for executing the given kernel function pointer \p kernel_ptr on the current device with \p block_threads per thread block.
 *
 * \par Snippet
 * The code snippet below illustrates the use of the MaxSmOccupancy function.
 * \par
 * \code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_device.cuh>
 *
 * template <typename T>
 * __global__ void ExampleKernel()
 * {
 *     // Allocate shared memory for BlockScan
 *     __shared__ volatile T buffer[4096];
 *
 *        ...
 * }
 *
 *     ...
 *
 * // Determine SM occupancy for ExampleKernel specialized for unsigned char
 * int max_sm_occupancy;
 * MaxSmOccupancy(max_sm_occupancy, ExampleKernel<unsigned char>, 64);
 *
 * // max_sm_occupancy  <-- 4 on SM10
 * // max_sm_occupancy  <-- 8 on SM20
 * // max_sm_occupancy  <-- 12 on SM35
 *
 * \endcode
 *
 */
template <typename KernelPtr>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t MaxSmOccupancy(
    int                 &max_sm_occupancy,          ///< [out] maximum number of thread blocks that can reside on a single SM
    KernelPtr           kernel_ptr,                 ///< [in] Kernel pointer for which to compute SM occupancy
    int                 block_threads)              ///< [in] Number of threads per thread block
{
#ifndef CUB_RUNTIME_ENABLED

    // CUDA API calls not supported from this device
    return CubDebug(cudaErrorInvalidConfiguration);

#else

    cudaError_t error = cudaSuccess;
    do
    {
        // Get device ordinal
        int device_ordinal;
        if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

        // Get device SM version
        int sm_version;
        if (CubDebug(error = SmVersion(sm_version, device_ordinal))) break;

        // Get SM occupancy
        if (CubDebug(error = MaxSmOccupancy(max_sm_occupancy, sm_version, kernel_ptr, block_threads))) break;

    } while (0);

    return error;

#endif  // CUB_RUNTIME_ENABLED

}


/** @} */       // end group UtilMgmt

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
