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
 * Static architectural properties by SM version.
 */

#pragma once

#include "util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup UtilMgmt
 * @{
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

/// CUB_PTX_ARCH reflects the PTX version targeted by the active compiler pass (or zero during the host pass).
#ifndef __CUDA_ARCH__
    #define CUB_PTX_ARCH 0
#else
    #define CUB_PTX_ARCH __CUDA_ARCH__
#endif

/// Whether or not the source targeted by the active compiler pass is allowed to  invoke device kernels or methods from the CUDA runtime API.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__>= 350 && defined(__CUDACC_RDC__))
    #define CUB_RUNTIME_ENABLED
    #define CUB_RUNTIME_FUNCTION __host__ __device__
#else
    #define CUB_RUNTIME_FUNCTION __host__
#endif

#endif // DOXYGEN_SHOULD_SKIP_THIS


/// Number of threads per warp (log)
#define CUB_LOG_WARP_THREADS(arch)                      \
	(5)

/// Number of threads per warp
#define CUB_WARP_THREADS(arch)                          \
    (1 << CUB_LOG_WARP_THREADS(arch))

/// Number of smem banks (log)
#define CUB_LOG_SMEM_BANKS(arch)                        \
    ((arch >= 200) ?                                    \
        (5) :                                           \
        (4))

/// Number of smem banks
#define CUB_SMEM_BANKS(arch)                            \
    (1 << CUB_LOG_SMEM_BANKS(arch))

/// Number of bytes per smem bank
#define CUB_SMEM_BANK_BYTES(arch)                       \
    (4)

/// Number of smem bytes provisioned per SM
#define CUB_SMEM_BYTES(arch)                            \
    ((arch >= 200) ?                                    \
		(48 * 1024) :                                   \
		(16 * 1024))

/// Smem allocation size in bytes
#define CUB_SMEM_ALLOC_UNIT(arch)                       \
    ((arch >= 300) ?                                    \
    	(256) :                                         \
		((arch >= 200) ?                                \
		    (128) :                                     \
		    (512)))

/// Whether or not the architecture allocates registers by block (or by warp)
#define CUB_REGS_BY_BLOCK(arch)                         \
    ((arch >= 200) ?                                    \
    	(false) :                                       \
    	(true))

/// Number of registers allocated at a time per block (or by warp)
#define CUB_REG_ALLOC_UNIT(arch)                        \
    ((arch >= 300) ?                                    \
    	(256) :                                         \
        ((arch >= 200) ?                                \
        	(64) :                                      \
            ((arch >= 120) ?                            \
            	(512) :                                 \
            	(256))))

/// Granularity of warps for which registers are allocated
#define CUB_WARP_ALLOC_UNIT(arch)                       \
    ((arch >= 300) ?                                    \
        (4) :                                           \
        (2))

/// Maximum number of threads per SM
#define CUB_MAX_SM_THREADS(arch)                        \
    ((arch >= 300) ?                                    \
    	(2048) :                                        \
        ((arch >= 200) ?                                \
        	(1536) :                                    \
            ((arch >= 120) ?                            \
           		(1024) :                                \
           		(768))))

/// Maximum number of thread blocks per SM
#define CUB_MAX_SM_BLOCKS(arch)                         \
    ((arch >= 300) ?                                    \
        (16) :                                          \
        (8))

/// Maximum number of threads per thread block
#define CUB_MAX_BLOCK_THREADS(arch)                     \
    ((arch >= 200) ?                                    \
        (1024) :                                        \
        (512))

/// Maximum number of registers per SM
#define CUB_MAX_SM_REGISTERS(arch)                      \
    ((arch >= 300) ?                                    \
        (64 * 1024) :                                   \
        ((arch >= 200) ?                                \
            (32 * 1024) :                               \
            ((arch >= 120) ?                            \
                (16 * 1024) :                           \
                (8 * 1024))))

/// Oversubscription factor
#define CUB_SUBSCRIPTION_FACTOR(arch)                   \
    ((arch >= 300) ?                                    \
        (5) :                                           \
        ((arch >= 200) ?                                \
            (3) :                                       \
            (10)))

/// Prefer padding overhead vs X-way conflicts greater than this threshold
#define CUB_PREFER_CONFLICT_OVER_PADDING(arch)          \
    ((arch >= 300) ?                                    \
        (1) :                                           \
        (4))

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

#define CUB_PTX_LOG_WARP_THREADS                CUB_LOG_WARP_THREADS(CUB_PTX_ARCH)
#define CUB_PTX_WARP_THREADS                    CUB_WARP_THREADS(CUB_PTX_ARCH)
#define CUB_PTX_LOG_SMEM_BANKS                  CUB_LOG_SMEM_BANKS(CUB_PTX_ARCH)
#define CUB_PTX_SMEM_BANKS                      CUB_SMEM_BANKS(CUB_PTX_ARCH)
#define CUB_PTX_SMEM_BANK_BYTES                 CUB_SMEM_BANK_BYTES(CUB_PTX_ARCH)
#define CUB_PTX_SMEM_BYTES                      CUB_SMEM_BYTES(CUB_PTX_ARCH)
#define CUB_PTX_SMEM_ALLOC_UNIT                 CUB_SMEM_ALLOC_UNIT(CUB_PTX_ARCH)
#define CUB_PTX_REGS_BY_BLOCK                   CUB_REGS_BY_BLOCK(CUB_PTX_ARCH)
#define CUB_PTX_REG_ALLOC_UNIT                  CUB_REG_ALLOC_UNIT(CUB_PTX_ARCH)
#define CUB_PTX_WARP_ALLOC_UNIT                 CUB_WARP_ALLOC_UNIT(CUB_PTX_ARCH)
#define CUB_PTX_MAX_SM_THREADS                  CUB_MAX_SM_THREADS(CUB_PTX_ARCH)
#define CUB_PTX_MAX_SM_BLOCKS                   CUB_MAX_SM_BLOCKS(CUB_PTX_ARCH)
#define CUB_PTX_MAX_BLOCK_THREADS               CUB_MAX_BLOCK_THREADS(CUB_PTX_ARCH)
#define CUB_PTX_MAX_SM_REGISTERS                CUB_MAX_SM_REGISTERS(CUB_PTX_ARCH)
#define CUB_PTX_PREFER_CONFLICT_OVER_PADDING    CUB_PREFER_CONFLICT_OVER_PADDING(CUB_PTX_ARCH)

#endif  // Do not document


/** @} */       // end group UtilMgmt

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
