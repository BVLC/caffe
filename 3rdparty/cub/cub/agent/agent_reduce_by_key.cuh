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
 * cub::AgentReduceByKey implements a stateful abstraction of CUDA thread blocks for participating in device-wide reduce-value-by-key.
 */

#pragma once

#include <iterator>

#include "single_pass_scan_operators.cuh"
#include "../block/block_load.cuh"
#include "../block/block_store.cuh"
#include "../block/block_scan.cuh"
#include "../block/block_discontinuity.cuh"
#include "../iterator/cache_modified_input_iterator.cuh"
#include "../iterator/constant_input_iterator.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentReduceByKey
 */
template <
    int                         _BLOCK_THREADS,                 ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,              ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,                ///< The BlockLoad algorithm to use
    CacheLoadModifier           _LOAD_MODIFIER,                 ///< Cache load modifier for reading input elements
    BlockScanAlgorithm          _SCAN_ALGORITHM>                ///< The BlockScan algorithm to use
struct AgentReduceByKeyPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,               ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,            ///< Items per thread (per tile of input)
    };

    static const BlockLoadAlgorithm     LOAD_ALGORITHM          = _LOAD_ALGORITHM;      ///< The BlockLoad algorithm to use
    static const CacheLoadModifier      LOAD_MODIFIER           = _LOAD_MODIFIER;       ///< Cache load modifier for reading input elements
    static const BlockScanAlgorithm     SCAN_ALGORITHM          = _SCAN_ALGORITHM;      ///< The BlockScan algorithm to use
};


/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief AgentReduceByKey implements a stateful abstraction of CUDA thread blocks for participating in device-wide reduce-value-by-key
 */
template <
    typename    AgentReduceByKeyPolicyT,        ///< Parameterized AgentReduceByKeyPolicy tuning policy type
    typename    KeysInputIteratorT,             ///< Random-access input iterator type for keys
    typename    UniqueOutputIteratorT,          ///< Random-access output iterator type for keys
    typename    ValuesInputIteratorT,           ///< Random-access input iterator type for values
    typename    AggregatesOutputIteratorT,      ///< Random-access output iterator type for values
    typename    NumRunsOutputIteratorT,         ///< Output iterator type for recording number of items selected
    typename    EqualityOpT,                    ///< KeyT equality operator type
    typename    ReductionOpT,                   ///< ValueT reduction operator type
    typename    OffsetT>                        ///< Signed integer type for global offsets
struct AgentReduceByKey
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    // Data type of key iterator
    typedef typename std::iterator_traits<KeysInputIteratorT>::value_type KeyT;

    // Data type of value iterator
    typedef typename std::iterator_traits<ValuesInputIteratorT>::value_type ValueT;

    // Tuple type for scanning (pairs accumulated segment-value with segment-index)
    typedef KeyValuePair<OffsetT, ValueT> OffsetValuePairT;

    // Tuple type for pairing keys and values
    typedef KeyValuePair<KeyT, ValueT> KeyValuePairT;

    // Tile status descriptor interface type
    typedef ReduceByKeyScanTileState<ValueT, OffsetT> ScanTileStateT;

    // Constants
    enum
    {
        BLOCK_THREADS       = AgentReduceByKeyPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD    = AgentReduceByKeyPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS          = BLOCK_THREADS * ITEMS_PER_THREAD,
        TWO_PHASE_SCATTER   = (ITEMS_PER_THREAD > 1),

        // Whether or not the scan operation has a zero-valued identity value (true if we're performing addition on a primitive type)
        HAS_IDENTITY_ZERO   = (Equals<ReductionOpT, cub::Sum>::VALUE) && (Traits<ValueT>::PRIMITIVE),
    };

    // Cache-modified Input iterator wrapper type (for applying cache modifier) for keys
    typedef typename If<IsPointer<KeysInputIteratorT>::VALUE,
            CacheModifiedInputIterator<AgentReduceByKeyPolicyT::LOAD_MODIFIER, KeyT, OffsetT>,      // Wrap the native input pointer with CacheModifiedValuesInputIterator
            KeysInputIteratorT>::Type                                                               // Directly use the supplied input iterator type
        WrappedKeysInputIteratorT;

    // Cache-modified Input iterator wrapper type (for applying cache modifier) for values
    typedef typename If<IsPointer<ValuesInputIteratorT>::VALUE,
            CacheModifiedInputIterator<AgentReduceByKeyPolicyT::LOAD_MODIFIER, ValueT, OffsetT>,    // Wrap the native input pointer with CacheModifiedValuesInputIterator
            ValuesInputIteratorT>::Type                                                             // Directly use the supplied input iterator type
        WrappedValuesInputIteratorT;

    // Cache-modified Input iterator wrapper type (for applying cache modifier) for fixup values
    typedef typename If<IsPointer<AggregatesOutputIteratorT>::VALUE,
            CacheModifiedInputIterator<AgentReduceByKeyPolicyT::LOAD_MODIFIER, ValueT, OffsetT>,    // Wrap the native input pointer with CacheModifiedValuesInputIterator
            AggregatesOutputIteratorT>::Type                                                        // Directly use the supplied input iterator type
        WrappedFixupInputIteratorT;

    // Reduce-value-by-segment scan operator
    typedef ReduceBySegmentOp<ReductionOpT> ReduceBySegmentOpT;

    // Parameterized BlockLoad type for keys
    typedef BlockLoad<
            WrappedKeysInputIteratorT,
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            AgentReduceByKeyPolicyT::LOAD_ALGORITHM>
        BlockLoadKeys;

    // Parameterized BlockLoad type for values
    typedef BlockLoad<
            WrappedValuesInputIteratorT,
            BLOCK_THREADS,
            ITEMS_PER_THREAD,
            AgentReduceByKeyPolicyT::LOAD_ALGORITHM>
        BlockLoadValues;

    // Parameterized BlockDiscontinuity type for keys
    typedef BlockDiscontinuity<
            KeyT,
            BLOCK_THREADS>
        BlockDiscontinuityKeys;

    // Parameterized BlockScan type
    typedef BlockScan<
            OffsetValuePairT,
            BLOCK_THREADS,
            AgentReduceByKeyPolicyT::SCAN_ALGORITHM>
        BlockScanT;

    // Callback type for obtaining tile prefix during block scan
    typedef TilePrefixCallbackOp<
            OffsetValuePairT,
            ReduceBySegmentOpT,
            ScanTileStateT>
        TilePrefixCallbackOpT;

    // Key and value exchange types
    typedef KeyT    KeyExchangeT[TILE_ITEMS + 1];
    typedef ValueT  ValueExchangeT[TILE_ITEMS + 1];

    // Shared memory type for this threadblock
    union _TempStorage
    {
        struct
        {
            typename BlockScanT::TempStorage                scan;           // Smem needed for tile scanning
            typename TilePrefixCallbackOpT::TempStorage     prefix;         // Smem needed for cooperative prefix callback
            typename BlockDiscontinuityKeys::TempStorage    discontinuity;  // Smem needed for discontinuity detection
        };

        // Smem needed for loading keys
        typename BlockLoadKeys::TempStorage load_keys;

        // Smem needed for loading values
        typename BlockLoadValues::TempStorage load_values;

        // Smem needed for compacting key value pairs(allows non POD items in this union)
        Uninitialized<KeyValuePairT[TILE_ITEMS + 1]> raw_exchange;
    };

    // Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    _TempStorage&                   temp_storage;       ///< Reference to temp_storage
    WrappedKeysInputIteratorT       d_keys_in;          ///< Input keys
    UniqueOutputIteratorT           d_unique_out;       ///< Unique output keys
    WrappedValuesInputIteratorT     d_values_in;        ///< Input values
    AggregatesOutputIteratorT       d_aggregates_out;   ///< Output value aggregates
    NumRunsOutputIteratorT          d_num_runs_out;     ///< Output pointer for total number of segments identified
    WrappedFixupInputIteratorT      d_fixup_in;         ///< Fixup input values
    InequalityWrapper<EqualityOpT>  inequality_op;      ///< KeyT inequality operator
    ReductionOpT                    reduction_op;       ///< Reduction operator
    ReduceBySegmentOpT              scan_op;            ///< Reduce-by-segment scan operator


    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    // Constructor
    __device__ __forceinline__
    AgentReduceByKey(
        TempStorage&                temp_storage,       ///< Reference to temp_storage
        KeysInputIteratorT          d_keys_in,          ///< Input keys
        UniqueOutputIteratorT       d_unique_out,       ///< Unique output keys
        ValuesInputIteratorT        d_values_in,        ///< Input values
        AggregatesOutputIteratorT   d_aggregates_out,   ///< Output value aggregates
        NumRunsOutputIteratorT      d_num_runs_out,     ///< Output pointer for total number of segments identified
        EqualityOpT                 equality_op,        ///< KeyT equality operator
        ReductionOpT                reduction_op)       ///< ValueT reduction operator
    :
        temp_storage(temp_storage.Alias()),
        d_keys_in(d_keys_in),
        d_unique_out(d_unique_out),
        d_values_in(d_values_in),
        d_aggregates_out(d_aggregates_out),
        d_num_runs_out(d_num_runs_out),
        d_fixup_in(d_aggregates_out),
        inequality_op(equality_op),
        reduction_op(reduction_op),
        scan_op(reduction_op)
    {}


    //---------------------------------------------------------------------
    // Block scan utility methods
    //---------------------------------------------------------------------

    /**
     * Scan with identity (first tile)
     */
    __device__ __forceinline__
    void ScanTile(
        OffsetValuePairT     (&scan_items)[ITEMS_PER_THREAD],
        OffsetValuePairT&    tile_aggregate,
        Int2Type<true>      has_identity)
    {
        OffsetValuePairT identity;
        identity.value = 0;
        identity.key = 0;
        BlockScanT(temp_storage.scan).ExclusiveScan(scan_items, scan_items, identity, scan_op, tile_aggregate);
    }

    /**
     * Scan without identity (first tile).  Without an identity, the first output item is undefined.
     *
     */
    __device__ __forceinline__
    void ScanTile(
        OffsetValuePairT     (&scan_items)[ITEMS_PER_THREAD],
        OffsetValuePairT&    tile_aggregate,
        Int2Type<false>     has_identity)
    {
        BlockScanT(temp_storage.scan).ExclusiveScan(scan_items, scan_items, scan_op, tile_aggregate);
    }

    /**
     * Scan with identity (subsequent tile)
     */
    __device__ __forceinline__
    void ScanTile(
        OffsetValuePairT             (&scan_items)[ITEMS_PER_THREAD],
        OffsetValuePairT&            tile_aggregate,
        TilePrefixCallbackOpT&      prefix_op,
        Int2Type<true>              has_identity)
    {
        OffsetValuePairT identity;
        identity.value = 0;
        identity.key = 0;
        BlockScanT(temp_storage.scan).ExclusiveScan(scan_items, scan_items, identity, scan_op, tile_aggregate, prefix_op);
    }

    /**
     * Scan without identity (subsequent tile).  Without an identity, the first output item is undefined.
     */
    __device__ __forceinline__
    void ScanTile(
        OffsetValuePairT             (&scan_items)[ITEMS_PER_THREAD],
        OffsetValuePairT&            tile_aggregate,
        TilePrefixCallbackOpT&      prefix_op,
        Int2Type<false>             has_identity)
    {
        BlockScanT(temp_storage.scan).ExclusiveScan(scan_items, scan_items, scan_op, tile_aggregate, prefix_op);
    }


    //---------------------------------------------------------------------
    // Zip utility methods
    //---------------------------------------------------------------------

    template <bool IS_LAST_TILE>
    __device__ __forceinline__ void ZipValuesAndFlags(
        OffsetT         num_remaining,
        ValueT          (&values)[ITEMS_PER_THREAD],
        OffsetT         (&segment_flags)[ITEMS_PER_THREAD],
        OffsetValuePairT (&scan_items)[ITEMS_PER_THREAD])
    {
        // Zip values and segment_flags
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Set segment_flags for first out-of-bounds item, zero for others
            if (IS_LAST_TILE && (OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM == num_remaining))
                segment_flags[ITEM] = 1;

            scan_items[ITEM].value      = values[ITEM];
            scan_items[ITEM].key     = segment_flags[ITEM];
        }
    }

    __device__ __forceinline__ void ZipKeysAndValues(
        KeyT            (&keys)[ITEMS_PER_THREAD],                  ///< in
        OffsetT         (&segment_indices)[ITEMS_PER_THREAD],       ///< out
        OffsetValuePairT   (&scan_items)[ITEMS_PER_THREAD],            ///< in
        KeyValuePairT   (&scatter_items)[ITEMS_PER_THREAD])         ///< out
    {
        // Zip values and segment_flags
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            scatter_items[ITEM].key     = keys[ITEM];
            scatter_items[ITEM].value   = scan_items[ITEM].value;
            segment_indices[ITEM]       = scan_items[ITEM].key;
        }
    }


    //---------------------------------------------------------------------
    // Scatter utility methods
    //---------------------------------------------------------------------

    /**
     * Directly scatter flagged items to output offsets (specialized for IS_SEGMENTED_REDUCTION_FIXUP == false)
     */
    __device__ __forceinline__ void ScatterDirect(
        KeyValuePairT   (&scatter_items)[ITEMS_PER_THREAD],
        OffsetT         (&segment_flags)[ITEMS_PER_THREAD],
        OffsetT         (&segment_indices)[ITEMS_PER_THREAD])
    {
        // Scatter flagged keys and values
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (segment_flags[ITEM])
            {
                // Scatter key
                d_unique_out[segment_indices[ITEM]] = scatter_items[ITEM].key;

                // Scatter value
                d_aggregates_out[segment_indices[ITEM]] = scatter_items[ITEM].value;
            }
        }
    }


    /**
     * 2-phase scatter flagged items to output offsets (specialized for IS_SEGMENTED_REDUCTION_FIXUP == false)
     *
     * The exclusive scan causes each head flag to be paired with the previous
     * value aggregate: the scatter offsets must be decremented for value aggregates
     */
    __device__ __forceinline__ void ScatterTwoPhase(
        KeyValuePairT   (&scatter_items)[ITEMS_PER_THREAD],
        OffsetT         (&segment_flags)[ITEMS_PER_THREAD],
        OffsetT         (&segment_indices)[ITEMS_PER_THREAD],
        OffsetT         num_tile_segments,
        OffsetT         num_tile_segments_prefix)
    {
        __syncthreads();

        // Compact and scatter keys
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (segment_flags[ITEM])
            {
                temp_storage.raw_exchange.Alias()[segment_indices[ITEM] - num_tile_segments_prefix] = scatter_items[ITEM];
            }
        }

        __syncthreads();

        for (int item = threadIdx.x; item < num_tile_segments; item += BLOCK_THREADS)
        {
            KeyValuePairT pair                                  = temp_storage.raw_exchange.Alias()[item];
            d_unique_out[num_tile_segments_prefix + item]       = pair.key;
            d_aggregates_out[num_tile_segments_prefix + item]   = pair.value;
        }
    }


    /**
     * Scatter flagged items
     */
    __device__ __forceinline__ void Scatter(
        KeyValuePairT   (&scatter_items)[ITEMS_PER_THREAD],
        OffsetT         (&segment_flags)[ITEMS_PER_THREAD],
        OffsetT         (&segment_indices)[ITEMS_PER_THREAD],
        OffsetT         num_tile_segments,
        OffsetT         num_tile_segments_prefix)
    {
        // Do a one-phase scatter if (a) two-phase is disabled or (b) the average number of selected items per thread is less than one
        if (TWO_PHASE_SCATTER && (num_tile_segments > BLOCK_THREADS))
        {
            ScatterTwoPhase(
                scatter_items,
                segment_flags,
                segment_indices,
                num_tile_segments,
                num_tile_segments_prefix);
        }
        else
        {
            ScatterDirect(
                scatter_items,
                segment_flags,
                segment_indices);
        }
    }


    //---------------------------------------------------------------------
    // Finalization utility methods
    //---------------------------------------------------------------------

    /**
     * Finalize the carry-out from the last tile (specialized for IS_SEGMENTED_REDUCTION_FIXUP == false)
     */
    __device__ __forceinline__ void FinalizeLastTile(
        OffsetT         num_segments,
        OffsetT         num_remaining,
        KeyT            last_key,
        ValueT          last_value)
    {
        // Last thread will output final count and last item, if necessary
        if (threadIdx.x == BLOCK_THREADS - 1)
        {
            // If the last tile is a whole tile, the inclusive prefix contains accumulated value reduction for the last segment
            if (num_remaining == TILE_ITEMS)
            {
                // Scatter key and value
                d_unique_out[num_segments] = last_key;
                d_aggregates_out[num_segments] = last_value;
                num_segments++;
            }

            // Output the total number of items selected
            *d_num_runs_out = num_segments;
        }
    }


    //---------------------------------------------------------------------
    // Cooperatively scan a device-wide sequence of tiles with other CTAs
    //---------------------------------------------------------------------


    /**
     * Process first tile of input (dynamic chained scan).  Returns the running count of segments and aggregated values (including this tile)
     */
    template <bool IS_LAST_TILE>
    __device__ __forceinline__ void ConsumeFirstTile(
        OffsetT             num_remaining,      ///< Number of global input items remaining (including this tile)
        OffsetT             tile_offset,        ///< Tile offset
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {
        KeyT                keys[ITEMS_PER_THREAD];             // Tile keys
        KeyT                pred_keys[ITEMS_PER_THREAD];        // Tile keys shifted up (predecessor)
        ValueT              values[ITEMS_PER_THREAD];           // Tile values
        OffsetT             segment_flags[ITEMS_PER_THREAD];    // Segment head flags
        OffsetT             segment_indices[ITEMS_PER_THREAD];  // Segment indices
        OffsetValuePairT     scan_items[ITEMS_PER_THREAD];       // Zipped values and segment flags|indices
        KeyValuePairT       scatter_items[ITEMS_PER_THREAD];    // Zipped key value pairs for scattering

        // Load keys (last tile repeats final element)
        if (IS_LAST_TILE)
            BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + tile_offset, keys, num_remaining);
        else
            BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + tile_offset, keys);

        __syncthreads();

        // Load values (last tile repeats final element)
        if (IS_LAST_TILE)
            BlockLoadValues(temp_storage.load_values).Load(d_values_in + tile_offset, values, num_remaining);
        else
            BlockLoadValues(temp_storage.load_values).Load(d_values_in + tile_offset, values);

        __syncthreads();

        // Set head segment_flags.  First tile sets the first flag for the first item
        BlockDiscontinuityKeys(temp_storage.discontinuity).FlagHeads(segment_flags, keys, pred_keys, inequality_op);

        // Unset the flag for the first item in the first tile so we won't scatter it
        if (threadIdx.x == 0)
            segment_flags[0] = 0;

        // Zip values and segment_flags
        ZipValuesAndFlags<IS_LAST_TILE>(num_remaining, values, segment_flags, scan_items);

        // Exclusive scan of values and segment_flags
        OffsetValuePairT tile_aggregate;
        ScanTile(scan_items, tile_aggregate, Int2Type<HAS_IDENTITY_ZERO>());

        if (threadIdx.x == 0)
        {
            // Update tile status if this is not the last tile
            if (!IS_LAST_TILE)
                tile_state.SetInclusive(0, tile_aggregate);

            // Initialize the segment index for the first scan item if necessary (the exclusive prefix for the first item is garbage)
            if (!HAS_IDENTITY_ZERO)
                scan_items[0].key = 0;
        }

        // Unzip values and segment indices
        ZipKeysAndValues(pred_keys, segment_indices, scan_items, scatter_items);

        // Scatter flagged items
        Scatter(
            scatter_items,
            segment_flags,
            segment_indices,
            tile_aggregate.key,
            0);

        if (IS_LAST_TILE)
        {
            // Finalize the carry-out from the last tile
            FinalizeLastTile(
                tile_aggregate.key,
                num_remaining,
                keys[ITEMS_PER_THREAD - 1],
                tile_aggregate.value);
        }
    }


    /**
     * Process subsequent tile of input (dynamic chained scan).  Returns the running count of segments and aggregated values (including this tile)
     */
    template <bool IS_LAST_TILE>
    __device__ __forceinline__ void ConsumeSubsequentTile(
        OffsetT             num_remaining,      ///< Number of global input items remaining (including this tile)
        int                 tile_idx,           ///< Tile index
        OffsetT             tile_offset,        ///< Tile offset
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {
        KeyT                keys[ITEMS_PER_THREAD];                 // Tile keys
        KeyT                pred_keys[ITEMS_PER_THREAD];            // Tile keys shifted up (predecessor)
        ValueT              values[ITEMS_PER_THREAD];               // Tile values
        OffsetT             segment_flags[ITEMS_PER_THREAD];        // Segment head flags
        OffsetT             segment_indices[ITEMS_PER_THREAD];      // Segment indices
        OffsetValuePairT     scan_items[ITEMS_PER_THREAD];           // Zipped values and segment flags|indices
        KeyValuePairT       scatter_items[ITEMS_PER_THREAD];    // Zipped key value pairs for scattering

        // Load keys (last tile repeats final element)
        if (IS_LAST_TILE)
            BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + tile_offset, keys, num_remaining);
        else
            BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in + tile_offset, keys);

        KeyT tile_pred_key = (threadIdx.x == 0) ?
            d_keys_in[tile_offset - 1] :
            ZeroInitialize<KeyT>();

        __syncthreads();

        // Load values (last tile repeats final element)
        if (IS_LAST_TILE)
            BlockLoadValues(temp_storage.load_values).Load(d_values_in + tile_offset, values, num_remaining);
        else
            BlockLoadValues(temp_storage.load_values).Load(d_values_in + tile_offset, values);

        __syncthreads();

        // Set head segment_flags
        BlockDiscontinuityKeys(temp_storage.discontinuity).FlagHeads(segment_flags, keys, pred_keys, inequality_op, tile_pred_key);

        // Zip values and segment_flags
        ZipValuesAndFlags<IS_LAST_TILE>(num_remaining, values, segment_flags, scan_items);

        // Exclusive scan of values and segment_flags
        OffsetValuePairT tile_aggregate;
        TilePrefixCallbackOpT prefix_op(tile_state, temp_storage.prefix, scan_op, tile_idx);
        ScanTile(scan_items, tile_aggregate, prefix_op, Int2Type<HAS_IDENTITY_ZERO>());
        OffsetValuePairT tile_inclusive_prefix = prefix_op.GetInclusivePrefix();

        // Unzip values and segment indices
        ZipKeysAndValues(pred_keys, segment_indices, scan_items, scatter_items);

        // Scatter flagged items
        Scatter(
            scatter_items,
            segment_flags,
            segment_indices,
            tile_aggregate.key,
            prefix_op.GetExclusivePrefix().key);

        if (IS_LAST_TILE)
        {
            // Finalize the carry-out from the last tile
            FinalizeLastTile(
                tile_inclusive_prefix.key,
                num_remaining,
                keys[ITEMS_PER_THREAD - 1],
                tile_inclusive_prefix.value);
        }
    }


    /**
     * Process a tile of input
     */
    template <
        bool                IS_LAST_TILE>
    __device__ __forceinline__ void ConsumeTile(
        OffsetT             num_remaining,      ///< Number of global input items remaining (including this tile)
        int                 tile_idx,           ///< Tile index
        OffsetT             tile_offset,        ///< Tile offset
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {

        if (tile_idx == 0)
        {
            ConsumeFirstTile<IS_LAST_TILE>(num_remaining, tile_offset, tile_state);
        }
        else
        {
            ConsumeSubsequentTile<IS_LAST_TILE>(num_remaining, tile_idx, tile_offset, tile_state);
        }
    }


    /**
     * Scan tiles of items as part of a dynamic chained scan
     */
    __device__ __forceinline__ void ConsumeRange(
        int                 num_items,          ///< Total number of input items
        int                 num_tiles,          ///< Total number of input tiles
        ScanTileStateT&     tile_state)         ///< Global tile state descriptor
    {
        // Blocks are launched in increasing order, so just assign one tile per block
        int     tile_idx        = (blockIdx.x * gridDim.y) + blockIdx.y;    // Current tile index
        OffsetT tile_offset     = tile_idx * TILE_ITEMS;                    // Global offset for the current tile
        OffsetT num_remaining   = num_items - tile_offset;                  // Remaining items (including this tile)

        if (num_remaining > TILE_ITEMS)
        {
            // Not the last tile (full)
            ConsumeTile<false>(num_remaining, tile_idx, tile_offset, tile_state);
        }
        else if (num_remaining > 0)
        {
            // The last tile (possibly partially-full)
            ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state);
        }
    }

};


}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

