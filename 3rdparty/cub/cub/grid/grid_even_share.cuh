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
 * cub::GridEvenShare is a descriptor utility for distributing input among CUDA threadblocks in an "even-share" fashion.  Each threadblock gets roughly the same number of fixed-size work units (grains).
 */


#pragma once

#include "../util_namespace.cuh"
#include "../util_macro.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup GridModule
 * @{
 */


/**
 * \brief GridEvenShare is a descriptor utility for distributing input among CUDA threadblocks in an "even-share" fashion.  Each threadblock gets roughly the same number of fixed-size work units (grains).
 *
 * \par Overview
 * GridEvenShare indicates which sections of input are to be mapped onto which threadblocks.
 * Threadblocks may receive one of three different amounts of work: "big", "normal",
 * and "last".  The "big" workloads are one scheduling grain larger than "normal".  The "last" work unit
 * for the last threadblock may be partially-full if the input is not an even multiple of
 * the scheduling grain size.
 *
 * \par
 * Before invoking a child grid, a parent thread will typically construct an instance of
 * GridEvenShare.  The instance can be passed to child threadblocks which can
 * initialize their per-threadblock offsets using \p BlockInit().
 *
 * \tparam OffsetT      Signed integer type for global offsets
 */
template <typename OffsetT>
struct GridEvenShare
{
    OffsetT     total_grains;
    int         big_blocks;
    OffsetT     big_share;
    OffsetT     normal_share;
    OffsetT     normal_base_offset;

    /// Total number of input items
    OffsetT     num_items;

    /// Grid size in threadblocks
    int         grid_size;

    /// OffsetT into input marking the beginning of the owning thread block's segment of input tiles
    OffsetT     block_offset;

    /// OffsetT into input of marking the end (one-past) of the owning thread block's segment of input tiles
    OffsetT     block_end;

    /**
     * \brief Default constructor.  Zero-initializes block-specific fields.
     */
    __host__ __device__ __forceinline__ GridEvenShare() :
        num_items(0),
        grid_size(0),
        block_offset(0),
        block_end(0) {}

    /**
     * \brief Constructor.  Initializes the grid-specific members \p num_items and \p grid_size. To be called prior prior to kernel launch)
     */
    __host__ __device__ __forceinline__ GridEvenShare(
        OffsetT  num_items,                 ///< Total number of input items
        int     max_grid_size,              ///< Maximum grid size allowable (actual grid size may be less if not warranted by the the number of input items)
        int     schedule_granularity)       ///< Granularity by which the input can be parcelled into and distributed among threablocks.  Usually the thread block's native tile size (or a multiple thereof.
    {
        this->num_items             = num_items;
        this->block_offset          = num_items;
        this->block_end             = num_items;
        this->total_grains          = (num_items + schedule_granularity - 1) / schedule_granularity;
        this->grid_size             = CUB_MIN(total_grains, max_grid_size);
        OffsetT grains_per_block     = total_grains / grid_size;
        this->big_blocks            = total_grains - (grains_per_block * grid_size);        // leftover grains go to big blocks
        this->normal_share          = grains_per_block * schedule_granularity;
        this->normal_base_offset    = big_blocks * schedule_granularity;
        this->big_share             = normal_share + schedule_granularity;
    }



    /**
     * \brief Initializes ranges for the specified partition index
     */
    __device__ __forceinline__ void Init(int partition_id)
    {
        if (partition_id < big_blocks)
        {
            // This threadblock gets a big share of grains (grains_per_block + 1)
            block_offset = (partition_id * big_share);
            block_end = block_offset + big_share;
        }
        else if (partition_id < total_grains)
        {
            // This threadblock gets a normal share of grains (grains_per_block)
            block_offset = normal_base_offset + (partition_id * normal_share);
            block_end = CUB_MIN(num_items, block_offset + normal_share);
        }
    }


    /**
     * \brief Initializes ranges for the current thread block (e.g., to be called by each threadblock after startup)
     */
    __device__ __forceinline__ void BlockInit()
    {
        Init(blockIdx.x);
    }


    /**
     * Print to stdout
     */
    __host__ __device__ __forceinline__ void Print()
    {
        printf(
#if (CUB_PTX_ARCH > 0)
            "\tthreadblock(%d) "
            "block_offset(%lu) "
            "block_end(%lu) "
#endif
            "num_items(%lu)  "
            "total_grains(%lu)  "
            "big_blocks(%lu)  "
            "big_share(%lu)  "
            "normal_share(%lu)\n",
#if (CUB_PTX_ARCH > 0)
                blockIdx.x,
                (unsigned long) block_offset,
                (unsigned long) block_end,
#endif
                (unsigned long) num_items,
                (unsigned long) total_grains,
                (unsigned long) big_blocks,
                (unsigned long) big_share,
                (unsigned long) normal_share);
    }
};



/** @} */       // end group GridModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
