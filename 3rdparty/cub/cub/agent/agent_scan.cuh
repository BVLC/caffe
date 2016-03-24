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
 * cub::AgentScan implements a stateful abstraction of CUDA thread blocks for participating in device-wide prefix scan .
 */

#pragma once

#include <iterator>

#include "single_pass_scan_operators.cuh"
#include "../block/block_load.cuh"
#include "../block/block_store.cuh"
#include "../block/block_scan.cuh"
#include "../grid/grid_queue.cuh"
#include "../iterator/cache_modified_input_iterator.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentScan
 */
template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    BlockStoreAlgorithm         _STORE_ALGORITHM,               ///< The BlockStore algorithm to use
    BlockScanAlgorithm          _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
struct AgentScanPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
    };

    static const BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;          ///< The BlockLoad algorithm to use
    static const CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;           ///< Cache load modifier for reading input elements
    static const BlockStoreAlgorithm    STORE_ALGORITHM         = _STORE_ALGORITHM;         ///< The BlockStore algorithm to use
    static const BlockScanAlgorithm     SCAN_ALGORITHM          = _SCAN_ALGORITHM;          ///< The BlockScan algorithm to use
};




/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief AgentScan implements a stateful abstraction of CUDA thread blocks for participating in device-wide prefix scan .
 */
template <
    typename AgentScanPolicyT,      ///< Parameterized AgentScanPolicyT tuning policy type
    typename InputIteratorT,        ///< Random-access input iterator type
    typename OutputIteratorT,       ///< Random-access output iterator type
    typename ScanOpT,               ///< Scan functor type
    typename IdentityT,             ///< The identity element for ScanOpT type (cub::NullType for inclusive scan)
    typename OffsetT>               ///< Signed integer type for global offsets
struct AgentScan
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of input iterator
    typedef typename std::iterator_traits<InputIteratorT>::value_type T;

    // Tile status descriptor interface type
    typedef ScanTileState<T> ScanTileStateT;

    // Input iterator wrapper type (for applying cache modifier)
    typedef typename If<IsPointer<InputIteratorT>::VALUE,
            CacheModifiedInputIterator<AgentScanPolicyT::LOAD_MODIFIER, T, OffsetT>,    // Wrap the native input pointer with CacheModifiedInputIterator
            InputIteratorT>::Type                                                            // Directly use the supplied input iterator type
        WrappedInputIteratorT;

    // Constants
    enum
    {
        INCLUSIVE           = Equals<IdentityT, NullType>::VALUE,            // Inclusive scan if no identity type is provided
        BLOCK_THREADS       = AgentScanPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD    = AgentScanPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,

        // Whether or not to sync after loading data
        SYNC_AFTER_LOAD     = (AgentScanPolicyT::LOAD_ALGORITHM != BLOCK_LOAD_DIRECT),
    };

    // Parameterized BlockLoad type
    typedef BlockLoad<
            WrappedInputIteratorT,
            AgentScanPolicyT::BLOCK_THREADS,
            AgentScanPolicyT::ITEMS_PER_THREAD,
            AgentScanPolicyT::LOAD_ALGORITHM>
        BlockLoadT;

    // Parameterized BlockStore type
    typedef BlockStore<
            OutputIteratorT,
            AgentScanPolicyT::BLOCK_THREADS,
            AgentScanPolicyT::ITEMS_PER_THREAD,
            AgentScanPolicyT::STORE_ALGORITHM>
        BlockStoreT;

    // Parameterized BlockScan type
    typedef BlockScan<
            T,
            AgentScanPolicyT::BLOCK_THREADS,
            AgentScanPolicyT::SCAN_ALGORITHM>
        BlockScanT;

    // Callback type for obtaining tile prefix during block scan
    typedef TilePrefixCallbackOp<
            T,
            ScanOpT,
            ScanTileStateT>
        TilePrefixCallbackOpT;

    // Stateful BlockScan prefix callback type for managing a running total while scanning consecutive tiles
    typedef BlockScanRunningPrefixOp<
            T,
            ScanOpT>
        RunningPrefixCallbackOp;

    // Shared memory type for this threadblock
    union _TempStorage
    {
        typename BlockLoadT::TempStorage    load;       // Smem needed for tile loading
        typename BlockStoreT::TempStorage   store;      // Smem needed for tile storing

        struct
        {
            typename TilePrefixCallbackOpT::TempStorage  prefix;     // Smem needed for cooperative prefix callback
            typename BlockScanT::TempStorage                scan;       // Smem needed for tile scanning
        };
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage&               temp_storage;       ///< Reference to temp_storage
    WrappedInputIteratorT       d_in;               ///< Input data
    OutputIteratorT             d_out;              ///< Output data
    ScanOpT                     scan_op;            ///< Binary scan operator
    IdentityT                   identity;           ///< The identity element for ScanOpT



    //---------------------------------------------------------------------
    // Block scan utility methods (first tile)
    //---------------------------------------------------------------------

    /**
     * Exclusive scan specialization
     */
    template <typename _ScanOp, typename _Identity>
    __device__ __forceinline__
    void ScanTile(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, _Identity identity, T& block_aggregate)
    {
        BlockScanT(temp_storage.scan).ExclusiveScan(items, items, identity, scan_op, block_aggregate);
    }

    /**
     * Exclusive sum specialization
     */
    template <typename _Identity>
    __device__ __forceinline__
    void ScanTile(T (&items)[ITEMS_PER_THREAD], Sum scan_op, _Identity identity, T& block_aggregate)
    {
        BlockScanT(temp_storage.scan).ExclusiveSum(items, items, block_aggregate);
    }

    /**
     * Inclusive scan specialization
     */
    template <typename _ScanOp>
    __device__ __forceinline__
    void ScanTile(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, NullType identity, T& block_aggregate)
    {
        BlockScanT(temp_storage.scan).InclusiveScan(items, items, scan_op, block_aggregate);
    }

    /**
     * Inclusive sum specialization
     */
    __device__ __forceinline__
    void ScanTile(T (&items)[ITEMS_PER_THREAD], Sum scan_op, NullType identity, T& block_aggregate)
    {
        BlockScanT(temp_storage.scan).InclusiveSum(items, items, block_aggregate);
    }

    //---------------------------------------------------------------------
    // Block scan utility methods (subsequent tiles)
    //---------------------------------------------------------------------

    /**
     * Exclusive scan specialization (with prefix from predecessors)
     */
    template <typename _ScanOp, typename _Identity, typename PrefixCallback>
    __device__ __forceinline__
    void ScanTile(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, _Identity identity, T& block_aggregate, PrefixCallback &prefix_op)
    {
        BlockScanT(temp_storage.scan).ExclusiveScan(items, items, identity, scan_op, block_aggregate, prefix_op);
    }

    /**
     * Exclusive sum specialization (with prefix from predecessors)
     */
    template <typename _Identity, typename PrefixCallback>
    __device__ __forceinline__
    void ScanTile(T (&items)[ITEMS_PER_THREAD], Sum scan_op, _Identity identity, T& block_aggregate, PrefixCallback &prefix_op)
    {
        BlockScanT(temp_storage.scan).ExclusiveSum(items, items, block_aggregate, prefix_op);
    }

    /**
     * Inclusive scan specialization (with prefix from predecessors)
     */
    template <typename _ScanOp, typename PrefixCallback>
    __device__ __forceinline__
    void ScanTile(T (&items)[ITEMS_PER_THREAD], _ScanOp scan_op, NullType identity, T& block_aggregate, PrefixCallback &prefix_op)
    {
        BlockScanT(temp_storage.scan).InclusiveScan(items, items, scan_op, block_aggregate, prefix_op);
    }

    /**
     * Inclusive sum specialization (with prefix from predecessors)
     */
    template <typename PrefixCallback>
    __device__ __forceinline__
    void ScanTile(T (&items)[ITEMS_PER_THREAD], Sum scan_op, NullType identity, T& block_aggregate, PrefixCallback &prefix_op)
    {
        BlockScanT(temp_storage.scan).InclusiveSum(items, items, block_aggregate, prefix_op);
    }


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    AgentScan(
        TempStorage&    temp_storage,       ///< Reference to temp_storage
        InputIteratorT  d_in,               ///< Input data
        OutputIteratorT d_out,              ///< Output data
        ScanOpT         scan_op,            ///< Binary scan operator
        IdentityT       identity)           ///< The identity element for ScanOpT
    :
        temp_storage(temp_storage.Alias()),
        d_in(d_in),
        d_out(d_out),
        scan_op(scan_op),
        identity(identity)
    {}


    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles with other CTAs
    //---------------------------------------------------------------------

    /**
     * Process a tile of input (dynamic chained scan)
     */
    template <bool IS_FULL_TILE>
    __device__ __forceinline__ void ConsumeTile(
        OffsetT             num_items,          ///< Total number of input items
        OffsetT             num_remaining,      ///< Total number of items remaining to be processed (including this tile)
        int                 tile_idx,           ///< Tile index
        OffsetT             tile_offset,        ///< Tile offset
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {
        // Load items
        T items[ITEMS_PER_THREAD];

        if (IS_FULL_TILE)
            BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items);
        else
            BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items, num_remaining);

        if (SYNC_AFTER_LOAD)
            __syncthreads();

        // Perform tile scan
        if (tile_idx == 0)
        {
            // Scan first tile
            T block_aggregate;
            ScanTile(items, scan_op, identity, block_aggregate);

            // Update tile status if there may be successor tiles (i.e., this tile is full)
            if (IS_FULL_TILE && (threadIdx.x == 0))
                tile_state.SetInclusive(0, block_aggregate);
        }
        else
        {
            // Scan non-first tile
            T block_aggregate;
            TilePrefixCallbackOpT prefix_op(tile_state, temp_storage.prefix, scan_op, tile_idx);
            ScanTile(items, scan_op, identity, block_aggregate, prefix_op);
        }

        __syncthreads();

        // Store items
        if (IS_FULL_TILE)
            BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items);
        else
            BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items, num_remaining);
    }


    /**
     * Dequeue and scan tiles of items as part of a dynamic chained scan
     */
    __device__ __forceinline__ void ConsumeRange(
        int                 num_items,          ///< Total number of input items
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {
        // Blocks are launched in increasing order, so just assign one tile per block
        int     tile_idx        = (blockIdx.x * gridDim.y) + blockIdx.y;   // Current tile index
        OffsetT tile_offset     = OffsetT(TILE_ITEMS) * tile_idx;          // Global offset for the current tile
        OffsetT num_remaining   = num_items - tile_offset;                 // Remaining items (including this tile)

        if (num_remaining > TILE_ITEMS)
        {
            // Full tile
            ConsumeTile<true>(num_items, num_remaining, tile_idx, tile_offset, tile_state);
        }
        else if (num_remaining > 0)
        {
            // Partially-full tile
            ConsumeTile<false>(num_items, num_remaining, tile_idx, tile_offset, tile_state);
        }
    }


    //---------------------------------------------------------------------
    // Scan an sequence of consecutive tiles (independent of other thread blocks)
    //---------------------------------------------------------------------

    /**
     * Process a tile of input
     */
    template <
        bool                        IS_FULL_TILE,
        bool                        IS_FIRST_TILE>
    __device__ __forceinline__ void ConsumeTile(
        OffsetT                     tile_offset,               ///< Tile offset
        RunningPrefixCallbackOp&    prefix_op,                  ///< Running prefix operator
        int                         valid_items = TILE_ITEMS)   ///< Number of valid items in the tile
    {
        // Load items
        T items[ITEMS_PER_THREAD];

        if (IS_FULL_TILE)
            BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items);
        else
            BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items, valid_items);

        __syncthreads();

        // Block scan
        if (IS_FIRST_TILE)
        {
            T block_aggregate;
            ScanTile(items, scan_op, identity, block_aggregate);
            prefix_op.running_total = block_aggregate;
        }
        else
        {
            T block_aggregate;
            ScanTile(items, scan_op, identity, block_aggregate, prefix_op);
        }

        __syncthreads();

        // Store items
        if (IS_FULL_TILE)
            BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items);
        else
            BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items, valid_items);
    }


    /**
     * Scan a consecutive share of input tiles
     */
    __device__ __forceinline__ void ConsumeRange(
        OffsetT  range_offset,      ///< [in] Threadblock begin offset (inclusive)
        OffsetT  range_end)         ///< [in] Threadblock end offset (exclusive)
    {
        BlockScanRunningPrefixOp<T, ScanOpT> prefix_op(scan_op);

        if (range_offset + TILE_ITEMS <= range_end)
        {
            // Consume first tile of input (full)
            ConsumeTile<true, true>(range_offset, prefix_op);
            range_offset += TILE_ITEMS;

            // Consume subsequent full tiles of input
            while (range_offset + TILE_ITEMS <= range_end)
            {
                ConsumeTile<true, false>(range_offset, prefix_op);
                range_offset += TILE_ITEMS;
            }

            // Consume a partially-full tile
            if (range_offset < range_end)
            {
                int valid_items = range_end - range_offset;
                ConsumeTile<false, false>(range_offset, prefix_op, valid_items);
            }
        }
        else
        {
            // Consume the first tile of input (partially-full)
            int valid_items = range_end - range_offset;
            ConsumeTile<false, true>(range_offset, prefix_op, valid_items);
        }
    }


    /**
     * Scan a consecutive share of input tiles, seeded with the specified prefix value
     */
    __device__ __forceinline__ void ConsumeRange(
        OffsetT range_offset,                       ///< [in] Threadblock begin offset (inclusive)
        OffsetT range_end,                          ///< [in] Threadblock end offset (exclusive)
        T       prefix)                             ///< [in] The prefix to apply to the scan segment
    {
        BlockScanRunningPrefixOp<T, ScanOpT> prefix_op(prefix, scan_op);

        // Consume full tiles of input
        while (range_offset + TILE_ITEMS <= range_end)
        {
            ConsumeTile<true, false>(range_offset, prefix_op);
            range_offset += TILE_ITEMS;
        }

        // Consume a partially-full tile
        if (range_offset < range_end)
        {
            int valid_items = range_end - range_offset;
            ConsumeTile<false, false>(range_offset, prefix_op, valid_items);
        }
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

