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
 * AgentRadixSortDownsweep implements a stateful abstraction of CUDA thread blocks for participating in device-wide radix sort downsweep .
 */


#pragma once

#include "../thread/thread_load.cuh"
#include "../block/block_load.cuh"
#include "../block/block_store.cuh"
#include "../block/block_radix_rank.cuh"
#include "../block/block_exchange.cuh"
#include "../util_type.cuh"
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
 * Types of scattering strategies
 */
enum RadixSortScatterAlgorithm
{
    RADIX_SORT_SCATTER_DIRECT,      ///< Scatter directly from registers to global bins
    RADIX_SORT_SCATTER_TWO_PHASE,   ///< First scatter from registers into shared memory bins, then into global bins
};


/**
 * Parameterizable tuning policy type for AgentRadixSortDownsweep
 */
template <
    int                         _BLOCK_THREADS,             ///< Threads per thread block
    int                         _ITEMS_PER_THREAD,          ///< Items per thread (per tile of input)
    BlockLoadAlgorithm          _LOAD_ALGORITHM,            ///< The BlockLoad algorithm to use
    CacheLoadModifier           _LOAD_MODIFIER,             ///< Cache load modifier for reading keys (and values)
    bool                        _MEMOIZE_OUTER_SCAN,        ///< Whether or not to buffer outer raking scan partials to incur fewer shared memory reads at the expense of higher register pressure.  See BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE for more details.
    BlockScanAlgorithm          _INNER_SCAN_ALGORITHM,      ///< The BlockScan algorithm algorithm to use
    RadixSortScatterAlgorithm   _SCATTER_ALGORITHM,         ///< The scattering strategy to use
    cudaSharedMemConfig         _SMEM_CONFIG,               ///< Shared memory bank mode
    int                         _RADIX_BITS>                ///< The number of radix bits, i.e., log2(bins)
struct AgentRadixSortDownsweepPolicy
{
    enum
    {
        BLOCK_THREADS           = _BLOCK_THREADS,           ///< Threads per thread block
        ITEMS_PER_THREAD        = _ITEMS_PER_THREAD,        ///< Items per thread (per tile of input)
        RADIX_BITS              = _RADIX_BITS,              ///< The number of radix bits, i.e., log2(bins)
        MEMOIZE_OUTER_SCAN      = _MEMOIZE_OUTER_SCAN,      ///< Whether or not to buffer outer raking scan partials to incur fewer shared memory reads at the expense of higher register pressure.  See BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE for more details.
    };

    static const BlockLoadAlgorithm         LOAD_ALGORITHM          = _LOAD_ALGORITHM;          ///< The BlockLoad algorithm to use
    static const CacheLoadModifier          LOAD_MODIFIER           = _LOAD_MODIFIER;           ///< Cache load modifier for reading keys (and values)
    static const BlockScanAlgorithm         INNER_SCAN_ALGORITHM    = _INNER_SCAN_ALGORITHM;    ///< The BlockScan algorithm algorithm to use
    static const RadixSortScatterAlgorithm  SCATTER_ALGORITHM       = _SCATTER_ALGORITHM;       ///< The scattering strategy to use
    static const cudaSharedMemConfig        SMEM_CONFIG             = _SMEM_CONFIG;             ///< Shared memory bank mode
};


/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * \brief AgentRadixSortDownsweep implements a stateful abstraction of CUDA thread blocks for participating in device-wide radix sort downsweep .
 */
template <
    typename AgentRadixSortDownsweepPolicy,             ///< Parameterized AgentRadixSortDownsweepPolicy tuning policy type
    bool     DESCENDING,                                ///< Whether or not the sorted-order is high-to-low
    typename KeyT,                                       ///< KeyT type
    typename ValueT,                                     ///< ValueT type
    typename OffsetT>                                   ///< Signed integer type for global offsets
struct AgentRadixSortDownsweep
{
    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    // Appropriate unsigned-bits representation of KeyT
    typedef typename Traits<KeyT>::UnsignedBits UnsignedBits;

    static const UnsignedBits MIN_KEY = Traits<KeyT>::MIN_KEY;
    static const UnsignedBits MAX_KEY = Traits<KeyT>::MAX_KEY;

    static const BlockLoadAlgorithm         LOAD_ALGORITHM          = AgentRadixSortDownsweepPolicy::LOAD_ALGORITHM;
    static const CacheLoadModifier          LOAD_MODIFIER           = AgentRadixSortDownsweepPolicy::LOAD_MODIFIER;
    static const BlockScanAlgorithm         INNER_SCAN_ALGORITHM    = AgentRadixSortDownsweepPolicy::INNER_SCAN_ALGORITHM;
    static const RadixSortScatterAlgorithm  SCATTER_ALGORITHM       = AgentRadixSortDownsweepPolicy::SCATTER_ALGORITHM;
    static const cudaSharedMemConfig        SMEM_CONFIG             = AgentRadixSortDownsweepPolicy::SMEM_CONFIG;

    enum
    {
        BLOCK_THREADS           = AgentRadixSortDownsweepPolicy::BLOCK_THREADS,
        ITEMS_PER_THREAD        = AgentRadixSortDownsweepPolicy::ITEMS_PER_THREAD,
        RADIX_BITS              = AgentRadixSortDownsweepPolicy::RADIX_BITS,
        MEMOIZE_OUTER_SCAN      = AgentRadixSortDownsweepPolicy::MEMOIZE_OUTER_SCAN,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,

        RADIX_DIGITS            = 1 << RADIX_BITS,
        KEYS_ONLY               = Equals<ValueT, NullType>::VALUE,

        WARP_THREADS            = CUB_PTX_LOG_WARP_THREADS,
        WARPS                   = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,

        BYTES_PER_SIZET         = sizeof(OffsetT),
        LOG_BYTES_PER_SIZET     = Log2<BYTES_PER_SIZET>::VALUE,

        LOG_SMEM_BANKS          = CUB_PTX_LOG_SMEM_BANKS,
        SMEM_BANKS              = 1 << LOG_SMEM_BANKS,

        DIGITS_PER_SCATTER_PASS = BLOCK_THREADS / SMEM_BANKS,
        SCATTER_PASSES          = RADIX_DIGITS / DIGITS_PER_SCATTER_PASS,

        LOG_STORE_TXN_THREADS   = LOG_SMEM_BANKS,
        STORE_TXN_THREADS       = 1 << LOG_STORE_TXN_THREADS,
    };

    // Input iterator wrapper type (for applying cache modifier)s
    typedef CacheModifiedInputIterator<LOAD_MODIFIER, UnsignedBits, OffsetT>    KeysItr;
    typedef CacheModifiedInputIterator<LOAD_MODIFIER, ValueT, OffsetT>          ValuesItr;

    // BlockRadixRank type
    typedef BlockRadixRank<
        BLOCK_THREADS,
        RADIX_BITS,
        DESCENDING,
        MEMOIZE_OUTER_SCAN,
        INNER_SCAN_ALGORITHM,
        SMEM_CONFIG> BlockRadixRank;

    // BlockLoad type (keys)
    typedef BlockLoad<
        KeysItr,
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        LOAD_ALGORITHM> BlockLoadKeys;

    // BlockLoad type (values)
    typedef BlockLoad<
        ValuesItr,
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        LOAD_ALGORITHM> BlockLoadValues;

    // BlockExchange type (keys)
    typedef BlockExchange<
        UnsignedBits,
        BLOCK_THREADS,
        ITEMS_PER_THREAD> BlockExchangeKeys;

    // BlockExchange type (values)
    typedef BlockExchange<
        ValueT,
        BLOCK_THREADS,
        ITEMS_PER_THREAD> BlockExchangeValues;


    /**
     * Shared memory storage layout
     */
    struct _TempStorage
    {
        OffsetT relative_bin_offsets[RADIX_DIGITS + 1];
        bool    short_circuit;

        union
        {
            typename BlockRadixRank::TempStorage        ranking;
            typename BlockLoadKeys::TempStorage         load_keys;
            typename BlockLoadValues::TempStorage       load_values;
            typename BlockExchangeKeys::TempStorage     exchange_keys;
            typename BlockExchangeValues::TempStorage   exchange_values;
        };
    };


    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    // Shared storage for this CTA
    _TempStorage    &temp_storage;

    // Input and output device pointers
    KeysItr         d_keys_in;
    ValuesItr       d_values_in;
    UnsignedBits    *d_keys_out;
    ValueT          *d_values_out;

    // The global scatter base offset for each digit (valid in the first RADIX_DIGITS threads)
    OffsetT         bin_offset;

    // The least-significant bit position of the current digit to extract
    int             current_bit;

    // Number of bits in current digit
    int             num_bits;

    // Whether to short-ciruit
    bool            short_circuit;



    //---------------------------------------------------------------------
    // Utility methods
    //---------------------------------------------------------------------

    /**
     * Decodes given keys to lookup digit offsets in shared memory
     */
    __device__ __forceinline__ void DecodeRelativeBinOffsets(
        UnsignedBits    (&twiddled_keys)[ITEMS_PER_THREAD],
        OffsetT         (&relative_bin_offsets)[ITEMS_PER_THREAD])
    {
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            UnsignedBits digit = BFE(twiddled_keys[KEY], current_bit, num_bits);

            // Lookup base digit offset from shared memory
            relative_bin_offsets[KEY] = temp_storage.relative_bin_offsets[digit];
        }
    }


    /**
     * Scatter ranked items to global memory
     */
    template <bool FULL_TILE, typename T>
    __device__ __forceinline__ void ScatterItems(
        T       (&items)[ITEMS_PER_THREAD],
        int     (&local_ranks)[ITEMS_PER_THREAD],
        OffsetT (&relative_bin_offsets)[ITEMS_PER_THREAD],
        T       *d_out,
        OffsetT valid_items)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            // Scatter if not out-of-bounds
            if (FULL_TILE || (local_ranks[ITEM] < valid_items))
            {
                d_out[relative_bin_offsets[ITEM] + local_ranks[ITEM]] = items[ITEM];
            }
        }
    }


    /**
     * Scatter ranked keys directly to global memory
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ScatterKeys(
        UnsignedBits                            (&twiddled_keys)[ITEMS_PER_THREAD],
        OffsetT                                 (&relative_bin_offsets)[ITEMS_PER_THREAD],
        int                                     (&ranks)[ITEMS_PER_THREAD],
        OffsetT                                 valid_items,
        Int2Type<RADIX_SORT_SCATTER_DIRECT>     scatter_algorithm)
    {
        // Compute scatter offsets
        DecodeRelativeBinOffsets(twiddled_keys, relative_bin_offsets);

        // Untwiddle keys before outputting
        UnsignedBits keys[ITEMS_PER_THREAD];

        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            keys[KEY] = Traits<KeyT>::TwiddleOut(twiddled_keys[KEY]);
        }

        // Scatter to global
        ScatterItems<FULL_TILE>(keys, ranks, relative_bin_offsets, d_keys_out, valid_items);
    }


    /**
     * Scatter ranked keys through shared memory, then to global memory
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ScatterKeys(
        UnsignedBits                            (&twiddled_keys)[ITEMS_PER_THREAD],
        OffsetT                                 (&relative_bin_offsets)[ITEMS_PER_THREAD],
        int                                     (&ranks)[ITEMS_PER_THREAD],
        OffsetT                                 valid_items,
        Int2Type<RADIX_SORT_SCATTER_TWO_PHASE>  scatter_algorithm)
    {
        // Exchange keys through shared memory
        BlockExchangeKeys(temp_storage.exchange_keys).ScatterToStriped(twiddled_keys, ranks);

        // Compute striped local ranks
        int local_ranks[ITEMS_PER_THREAD];

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            local_ranks[ITEM] = threadIdx.x + (ITEM * BLOCK_THREADS);
        }

        // Scatter directly
        ScatterKeys<FULL_TILE>(
            twiddled_keys,
            relative_bin_offsets,
            local_ranks,
            valid_items,
            Int2Type<RADIX_SORT_SCATTER_DIRECT>());
    }


    /**
     * Scatter ranked values directly to global memory
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ScatterValues(
        ValueT                                  (&values)[ITEMS_PER_THREAD],
        OffsetT                                 (&relative_bin_offsets)[ITEMS_PER_THREAD],
        int                                     (&ranks)[ITEMS_PER_THREAD],
        OffsetT                                 valid_items,
        Int2Type<RADIX_SORT_SCATTER_DIRECT>     scatter_algorithm)
    {
        // Scatter to global
        ScatterItems<FULL_TILE>(values, ranks, relative_bin_offsets, d_values_out, valid_items);
    }


    /**
     * Scatter ranked values through shared memory, then to global memory
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ScatterValues(
        ValueT                                  (&values)[ITEMS_PER_THREAD],
        OffsetT                                 (&relative_bin_offsets)[ITEMS_PER_THREAD],
        int                                     (&ranks)[ITEMS_PER_THREAD],
        OffsetT                                 valid_items,
        Int2Type<RADIX_SORT_SCATTER_TWO_PHASE>  scatter_algorithm)
    {
        __syncthreads();

        // Exchange keys through shared memory
        BlockExchangeValues(temp_storage.exchange_values).ScatterToStriped(values, ranks);

        // Compute striped local ranks
        int local_ranks[ITEMS_PER_THREAD];

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            local_ranks[ITEM] = threadIdx.x + (ITEM * BLOCK_THREADS);
        }

        // Scatter directly
        ScatterValues<FULL_TILE>(
            values,
            relative_bin_offsets,
            local_ranks,
            valid_items,
            Int2Type<RADIX_SORT_SCATTER_DIRECT>());
    }


    /**
     * Load a tile of items (specialized for full tile)
     */
    template <typename BlockLoadT, typename T, typename InputIteratorT>
    __device__ __forceinline__ void LoadItems(
        BlockLoadT      &block_loader, 
        T               (&items)[ITEMS_PER_THREAD],
        InputIteratorT  d_in,
        OffsetT         valid_items,
        Int2Type<true>  is_full_tile)
    {
        block_loader.Load(d_in, items);
    }


    /**
     * Load a tile of items (specialized for full tile)
     */
    template <typename BlockLoadT, typename T, typename InputIteratorT>
    __device__ __forceinline__ void LoadItems(
        BlockLoadT      &block_loader,
        T               (&items)[ITEMS_PER_THREAD],
        InputIteratorT  d_in,
        OffsetT         valid_items,
        T               oob_item,
        Int2Type<true>  is_full_tile)
    {
        block_loader.Load(d_in, items);
    }


    /**
     * Load a tile of items (specialized for partial tile)
     */
    template <typename BlockLoadT, typename T, typename InputIteratorT>
    __device__ __forceinline__ void LoadItems(
        BlockLoadT      &block_loader, 
        T               (&items)[ITEMS_PER_THREAD],
        InputIteratorT  d_in,
        OffsetT         valid_items,
        Int2Type<false> is_full_tile)
    {
        block_loader.Load(d_in, items, valid_items);
    }

    /**
     * Load a tile of items (specialized for partial tile)
     */
    template <typename BlockLoadT, typename T, typename InputIteratorT>
    __device__ __forceinline__ void LoadItems(
        BlockLoadT      &block_loader,
        T               (&items)[ITEMS_PER_THREAD],
        InputIteratorT  d_in,
        OffsetT         valid_items,
        T               oob_item,
        Int2Type<false> is_full_tile)
    {
        block_loader.Load(d_in, items, valid_items, oob_item);
    }


    /**
     * Truck along associated values
     */
    template <bool FULL_TILE, typename _ValueT>
    __device__ __forceinline__ void GatherScatterValues(
        _ValueT     (&values)[ITEMS_PER_THREAD],
        OffsetT     (&relative_bin_offsets)[ITEMS_PER_THREAD],
        int         (&ranks)[ITEMS_PER_THREAD],
        OffsetT     block_offset,
        OffsetT     valid_items)
    {
        __syncthreads();

        BlockLoadValues loader(temp_storage.load_values);
        LoadItems(
            loader,
            values,
            d_values_in + block_offset,
            valid_items,
            Int2Type<FULL_TILE>());

        ScatterValues<FULL_TILE>(
            values,
            relative_bin_offsets,
            ranks,
            valid_items,
            Int2Type<SCATTER_ALGORITHM>());
    }


    /**
     * Truck along associated values (specialized for key-only sorting)
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void GatherScatterValues(
        NullType    (&values)[ITEMS_PER_THREAD],
        OffsetT     (&relative_bin_offsets)[ITEMS_PER_THREAD],
        int         (&ranks)[ITEMS_PER_THREAD],
        OffsetT     block_offset,
        OffsetT     valid_items)
    {}


    /**
     * Process tile
     */
    template <bool FULL_TILE>
    __device__ __forceinline__ void ProcessTile(
        OffsetT block_offset,
        const OffsetT &valid_items = TILE_ITEMS)
    {
        // Per-thread tile data
        UnsignedBits    keys[ITEMS_PER_THREAD];                     // Keys
        UnsignedBits    twiddled_keys[ITEMS_PER_THREAD];            // Twiddled keys
        int             ranks[ITEMS_PER_THREAD];                    // For each key, the local rank within the CTA
        OffsetT         relative_bin_offsets[ITEMS_PER_THREAD];     // For each key, the global scatter base offset of the corresponding digit

        // Assign default (min/max) value to all keys
        UnsignedBits default_key = (DESCENDING) ? MIN_KEY : MAX_KEY;

        // Load tile of keys
        BlockLoadKeys loader(temp_storage.load_keys);
        LoadItems(
            loader,
            keys,
            d_keys_in + block_offset,
            valid_items, 
            default_key,
            Int2Type<FULL_TILE>());

        __syncthreads();

        // Twiddle key bits if necessary
        #pragma unroll
        for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
        {
            twiddled_keys[KEY] = Traits<KeyT>::TwiddleIn(keys[KEY]);
        }

        // Rank the twiddled keys
        int inclusive_digit_prefix;
        BlockRadixRank(temp_storage.ranking).RankKeys(
            twiddled_keys,
            ranks,
            current_bit,
            num_bits,
            inclusive_digit_prefix);

        // Update global scatter base offsets for each digit
        if ((BLOCK_THREADS == RADIX_DIGITS) || (threadIdx.x < RADIX_DIGITS))
        {
            int exclusive_digit_prefix;

            // Get exclusive digit prefix from inclusive prefix
            if (DESCENDING)
            {
                // Get the prefix from the next thread (higher bins come first)
#if CUB_PTX_ARCH >= 300
                exclusive_digit_prefix = ShuffleDown(inclusive_digit_prefix, 1);
                if (threadIdx.x == RADIX_DIGITS - 1)
                    exclusive_digit_prefix = 0;
#else
                volatile int* exchange = reinterpret_cast<int *>(temp_storage.relative_bin_offsets);
                exchange[threadIdx.x + 1] = 0;
                exchange[threadIdx.x] = inclusive_digit_prefix;
                exclusive_digit_prefix = exchange[threadIdx.x + 1];
#endif
            }
            else
            {
                // Get the prefix from the previous thread (lower bins come first)
#if CUB_PTX_ARCH >= 300
                exclusive_digit_prefix = ShuffleUp(inclusive_digit_prefix, 1);
                if (threadIdx.x == 0)
                    exclusive_digit_prefix = 0;
#else
                volatile int* exchange = reinterpret_cast<int *>(temp_storage.relative_bin_offsets);
                exchange[threadIdx.x] = 0;
                exchange[threadIdx.x + 1] = inclusive_digit_prefix;
                exclusive_digit_prefix = exchange[threadIdx.x];
#endif
            }

            bin_offset -= exclusive_digit_prefix;
            temp_storage.relative_bin_offsets[threadIdx.x] = bin_offset;
            bin_offset += inclusive_digit_prefix;
        }

        __syncthreads();

        // Scatter keys
        ScatterKeys<FULL_TILE>(twiddled_keys, relative_bin_offsets, ranks, valid_items, Int2Type<SCATTER_ALGORITHM>());

        // Gather/scatter values
        ValueT values[ITEMS_PER_THREAD];
        GatherScatterValues<FULL_TILE>(values, relative_bin_offsets, ranks, block_offset, valid_items);
    }

    //---------------------------------------------------------------------
    // Copy shortcut
    //---------------------------------------------------------------------

    /**
     * Copy tiles within the range of input
     */
    template <
        typename InputIteratorT,
        typename T>
    __device__ __forceinline__ void Copy(
        InputIteratorT  d_in,
        T               *d_out,
        OffsetT         block_offset,
        OffsetT         block_end)
    {
        // Simply copy the input
        while (block_offset + TILE_ITEMS <= block_end)
        {
            T items[ITEMS_PER_THREAD];

            LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_in + block_offset, items);
            __syncthreads();
            StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + block_offset, items);

            block_offset += TILE_ITEMS;
        }

        // Clean up last partial tile with guarded-I/O
        if (block_offset < block_end)
        {
            OffsetT valid_items = block_end - block_offset;

            T items[ITEMS_PER_THREAD];

            LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_in + block_offset, items, valid_items);
            __syncthreads();
            StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + block_offset, items, valid_items);
        }
    }


    /**
     * Copy tiles within the range of input (specialized for NullType)
     */
    template <typename InputIteratorT>
    __device__ __forceinline__ void Copy(
        InputIteratorT  d_in,
        NullType        *d_out,
        OffsetT         block_offset,
        OffsetT         block_end)
    {}


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ AgentRadixSortDownsweep(
        TempStorage &temp_storage,
        OffsetT      bin_offset,
        KeyT        *d_keys_in,
        KeyT        *d_keys_out,
        ValueT      *d_values_in,
        ValueT      *d_values_out,
        int         current_bit,
        int         num_bits)
    :
        temp_storage(temp_storage.Alias()),
        bin_offset(bin_offset),
        d_keys_in(reinterpret_cast<UnsignedBits*>(d_keys_in)),
        d_keys_out(reinterpret_cast<UnsignedBits*>(d_keys_out)),
        d_values_in(d_values_in),
        d_values_out(d_values_out),
        current_bit(current_bit),
        num_bits(num_bits),
        short_circuit(false)
    {}


    /**
     * Constructor
     */
    __device__ __forceinline__ AgentRadixSortDownsweep(
        TempStorage &temp_storage,
        OffsetT     num_items,
        OffsetT     *d_spine,
        KeyT        *d_keys_in,
        KeyT        *d_keys_out,
        ValueT      *d_values_in,
        ValueT      *d_values_out,
        int         current_bit,
        int         num_bits)
    :
        temp_storage(temp_storage.Alias()),
        d_keys_in(reinterpret_cast<UnsignedBits*>(d_keys_in)),
        d_keys_out(reinterpret_cast<UnsignedBits*>(d_keys_out)),
        d_values_in(d_values_in),
        d_values_out(d_values_out),
        current_bit(current_bit),
        num_bits(num_bits)
    {
        // Load digit bin offsets (each of the first RADIX_DIGITS threads will load an offset for that digit)
        if (threadIdx.x < RADIX_DIGITS)
        {
            int bin_idx = (DESCENDING) ?
                RADIX_DIGITS - threadIdx.x - 1 :
                threadIdx.x;

            // Short circuit if the first block's histogram has only bin counts of only zeros or problem-size
            OffsetT first_block_bin_offset = d_spine[gridDim.x * bin_idx];
            int predicate = ((first_block_bin_offset == 0) || (first_block_bin_offset == num_items));
            this->temp_storage.short_circuit = WarpAll(predicate);

            // Load my block's bin offset for my bin
            bin_offset = d_spine[(gridDim.x * bin_idx) + blockIdx.x];
        }

        __syncthreads();

        short_circuit = this->temp_storage.short_circuit;
    }


    /**
     * Distribute keys from a segment of input tiles.
     */
    __device__ __forceinline__ void ProcessRegion(
        OffsetT         block_offset,
        const OffsetT   &block_end)
    {
        if (short_circuit)
        {
            // Copy keys
            Copy(d_keys_in, d_keys_out, block_offset, block_end);

            // Copy values
            Copy(d_values_in, d_values_out, block_offset, block_end);
        }
        else
        {
            // Process full tiles of tile_items
            while (block_offset + TILE_ITEMS <= block_end)
            {
                ProcessTile<true>(block_offset);
                block_offset += TILE_ITEMS;

                __syncthreads();
            }

            // Clean up last partial tile with guarded-I/O
            if (block_offset < block_end)
            {
                ProcessTile<false>(block_offset, block_end - block_offset);
            }
        }
    }

};



}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)

