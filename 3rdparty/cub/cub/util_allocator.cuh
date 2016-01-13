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

/******************************************************************************
 * Simple caching allocator for device memory allocations. The allocator is
 * thread-safe and capable of managing device allocations on multiple devices.
 ******************************************************************************/

#pragma once

#if (CUB_PTX_ARCH == 0)
    #include <set>              // NVCC (EDG, really) takes FOREVER to compile std::map
    #include <map>
#endif

#include <math.h>

#include "util_namespace.cuh"
#include "util_debug.cuh"

#include "host/spinlock.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \addtogroup UtilMgmt
 * @{
 */


/******************************************************************************
 * CachingDeviceAllocator (host use)
 ******************************************************************************/

/**
 * \brief A simple caching allocator for device memory allocations.
 *
 * \par Overview
 * The allocator is thread-safe and stream-safe and is capable of managing cached
 * device allocations on multiple devices.  It behaves as follows:
 *
 * \par
 * - Allocations from the allocator are associated with an \p active_stream.  Once freed,
 *   the allocation becomes available immediately for reuse within the \p active_stream
 *   with which it was associated with during allocation, and it becomes available for
 *   reuse within other streams when all prior work submitted to \p active_stream has completed.
 * - Allocations are categorized and cached by bin size.  A new allocation request of
 *   a given size will only consider cached allocations within the corresponding bin.
 * - Bin limits progress geometrically in accordance with the growth factor
 *   \p bin_growth provided during construction.  Unused device allocations within
 *   a larger bin cache are not reused for allocation requests that categorize to
 *   smaller bin sizes.
 * - Allocation requests below (\p bin_growth ^ \p min_bin) are rounded up to
 *   (\p bin_growth ^ \p min_bin).
 * - Allocations above (\p bin_growth ^ \p max_bin) are not rounded up to the nearest
 *   bin and are simply freed when they are deallocated instead of being returned
 *   to a bin-cache.
 * - %If the total storage of cached allocations on a given device will exceed
 *   \p max_cached_bytes, allocations for that device are simply freed when they are
 *   deallocated instead of being returned to their bin-cache.
 *
 * \par
 * For example, the default-constructed CachingDeviceAllocator is configured with:
 * - \p bin_growth = 8
 * - \p min_bin = 3
 * - \p max_bin = 7
 * - \p max_cached_bytes = 6MB - 1B
 *
 * \par
 * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB
 * and sets a maximum of 6,291,455 cached bytes per device
 *
 */
struct CachingDeviceAllocator
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


    //---------------------------------------------------------------------
    // Type definitions and constants
    //---------------------------------------------------------------------

    enum
    {
        /// Invalid device ordinal
        INVALID_DEVICE_ORDINAL  = -1,
    };

    /**
     * Integer pow function for unsigned base and exponent
     */
    static unsigned int IntPow(
        unsigned int base,
        unsigned int exp)
    {
        unsigned int retval = 1;
        while (exp > 0)
        {
            if (exp & 1) {
                retval = retval * base;        // multiply the result by the current base
            }
            base = base * base;                // square the base
            exp = exp >> 1;                    // divide the exponent in half
        }
        return retval;
    }


    /**
     * Round up to the nearest power-of
     */
    static void NearestPowerOf(
        unsigned int &power,
        size_t &rounded_bytes,
        unsigned int base,
        size_t value)
    {
        power = 0;
        rounded_bytes = 1;

        while (rounded_bytes < value)
        {
            rounded_bytes *= base;
            power++;
        }
    }

    /**
     * Descriptor for device memory allocations
     */
    struct BlockDescriptor
    {
        void*           d_ptr;              // Device pointer
        size_t          bytes;              // Size of allocation in bytes
        unsigned int    bin;                // Bin enumeration
        int             device;             // device ordinal
        cudaStream_t    associated_stream;  // Associated associated_stream
        cudaEvent_t     ready_event;        // Signal when associated stream has run to the point at which this block was freed

        // Constructor
        BlockDescriptor(void *d_ptr, int device) :
            d_ptr(d_ptr),
            bytes(0),
            bin(0),
            device(device),
            associated_stream(0),
            ready_event(0)
        {}

        // Constructor
        BlockDescriptor(size_t bytes, unsigned int bin, int device, cudaStream_t associated_stream) :
            d_ptr(NULL),
            bytes(bytes),
            bin(bin),
            device(device),
            associated_stream(associated_stream),
            ready_event(0)
        {}

        // Comparison functor for comparing device pointers
        static bool PtrCompare(const BlockDescriptor &a, const BlockDescriptor &b)
        {
            if (a.device == b.device)
                return (a.d_ptr < b.d_ptr);
            else
                return (a.device < b.device);
        }

        // Comparison functor for comparing allocation sizes
        static bool SizeCompare(const BlockDescriptor &a, const BlockDescriptor &b)
        {
            if (a.device == b.device)
                return (a.bytes < b.bytes);
            else
                return (a.device < b.device);
        }
    };

    /// BlockDescriptor comparator function interface
    typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);

#if (CUB_PTX_ARCH == 0)   // Only define STL container members in host code

    class TotalBytes {
    public:
      size_t free;
      size_t busy;
      TotalBytes() { free = busy = 0; }
    };

    /// Set type for cached blocks (ordered by size)
    typedef std::multiset<BlockDescriptor, Compare> CachedBlocks;

    /// Set type for live blocks (ordered by ptr)
    typedef std::multiset<BlockDescriptor, Compare> BusyBlocks;

    /// Map type of device ordinals to the number of cached bytes cached by each device
  typedef std::map<int, TotalBytes> GpuCachedBytes;

#endif // CUB_PTX_ARCH

    //---------------------------------------------------------------------
    // Fields
    //---------------------------------------------------------------------

    Spinlock        spin_lock;          /// Spinlock for thread-safety

    unsigned int    bin_growth;         /// Geometric growth factor for bin-sizes
    unsigned int    min_bin;            /// Minimum bin enumeration
    unsigned int    max_bin;            /// Maximum bin enumeration

    size_t          min_bin_bytes;      /// Minimum bin size
    size_t          max_bin_bytes;      /// Maximum bin size
    size_t          max_cached_bytes;   /// Maximum aggregate cached bytes per device

    const bool      skip_cleanup;       /// Whether or not to skip a call to FreeAllCached() when destructor is called.  (The CUDA runtime may have already shut down for statically declared allocators)
    bool            debug;              /// Whether or not to print (de)allocation events to stdout

#if (CUB_PTX_ARCH == 0)   // Only define STL container members in host code

    GpuCachedBytes  cached_bytes;       /// Map of device ordinal to aggregate cached bytes on that device
    CachedBlocks    cached_blocks;      /// Set of cached device allocations available for reuse
    BusyBlocks      live_blocks;        /// Set of live device allocations currently in use

#endif // CUB_PTX_ARCH

#endif // DOXYGEN_SHOULD_SKIP_THIS

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * \brief Constructor.
     */
    CachingDeviceAllocator(
        unsigned int    bin_growth,             ///< Geometric growth factor for bin-sizes
        unsigned int    min_bin,                ///< Minimum bin
        unsigned int    max_bin,                ///< Maximum bin
        size_t          max_cached_bytes,       ///< Maximum aggregate cached bytes per device
        bool            skip_cleanup = false,   ///< Whether or not to skip a call to \p FreeAllCached() when the destructor is called.
        bool            debug = false           ///<  Whether or not to print (de)allocation events to stdout
        )
    :
            spin_lock(0),
            bin_growth(bin_growth),
            min_bin(min_bin),
            max_bin(max_bin),
            min_bin_bytes(IntPow(bin_growth, min_bin)),
            max_bin_bytes(IntPow(bin_growth, max_bin)),
            max_cached_bytes(max_cached_bytes),
            skip_cleanup(skip_cleanup),
            debug(debug)
    #if (CUB_PTX_ARCH == 0)   // Only define STL container members in host code
            ,cached_blocks(BlockDescriptor::SizeCompare)
            ,live_blocks(BlockDescriptor::PtrCompare)
    #endif
    {}


    /**
     * \brief Default constructor.
     *
     * Configured with:
     * \par
     * - \p bin_growth = 8
     * - \p min_bin = 3
     * - \p max_bin = 7
     * - \p max_cached_bytes = (\p bin_growth ^ \p max_bin) * 3) - 1 = 6,291,455 bytes
     *
     * which delineates five bin-sizes: 512B, 4KB, 32KB, 256KB, and 2MB and
     * sets a maximum of 6,291,455 cached bytes per device
     */
    CachingDeviceAllocator(
        bool skip_cleanup = false,
        bool debug = false)
    :
        spin_lock(0),
        bin_growth(8),
        min_bin(3),
        max_bin(7),
        min_bin_bytes(IntPow(bin_growth, min_bin)),
        max_bin_bytes(IntPow(bin_growth, max_bin)),
        max_cached_bytes((max_bin_bytes * 3) - 1),
        skip_cleanup(skip_cleanup),
        debug(debug)
    #if (CUB_PTX_ARCH == 0)   // Only define STL container members in host code
        ,cached_blocks(BlockDescriptor::SizeCompare)
        ,live_blocks(BlockDescriptor::PtrCompare)
    #endif
    {}


    /**
     * \brief Sets the limit on the number bytes this allocator is allowed to cache per device.
     */
    cudaError_t SetMaxCachedBytes(
        size_t max_cached_bytes)
    {
    #if (CUB_PTX_ARCH > 0)
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        // Lock
        Lock(&spin_lock);

        this->max_cached_bytes = max_cached_bytes;

        if (debug) CubLog("New max_cached_bytes(%lld)\n", (long long) max_cached_bytes);

        // Unlock
        Unlock(&spin_lock);

        return cudaSuccess;

    #endif // CUB_PTX_ARCH
    }


    /**
     * \brief Provides a suitable allocation of device memory for the given size on the specified device.
     *
     * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
     * with which it was associated with during allocation, and it becomes available for reuse within other
     * streams when all prior work submitted to \p active_stream has completed.
     */
    cudaError_t DeviceAllocate(
        int             device,             ///< [in] Device on which to place the allocation
        void            **d_ptr,            ///< [out] Reference to pointer to the allocation
        size_t          bytes,              ///< [in] Minimum number of bytes for the allocation
        cudaStream_t    active_stream = 0)  ///< [in] The stream to be associated with this allocation
    {
    #if (CUB_PTX_ARCH > 0)
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        *d_ptr                          = NULL;

        int entrypoint_device = INVALID_DEVICE_ORDINAL;
        cudaError_t error               = cudaSuccess;

        if (device == INVALID_DEVICE_ORDINAL) {
	 if (CubDebug(error = cudaGetDevice(&entrypoint_device)))
	   return error;
         device = entrypoint_device;
	}

        // Round up to nearest bin size
        unsigned int bin;
        size_t bin_bytes;
        NearestPowerOf(bin, bin_bytes, bin_growth, bytes);
        if (bin < min_bin) {
          bin = min_bin;
          bin_bytes = min_bin_bytes;
        }

        // Check if bin is greater than our maximum bin
        if (bin > max_bin)
          {
            // Allocate the request exactly and give out-of-range bin
            bin = (unsigned int) -1;
            bin_bytes = bytes;
          }

        BlockDescriptor search_key(bin_bytes, bin, device, active_stream);

        // Lock while we search
        Lock(&spin_lock);

        // Find the range of freed blocks big enough within the same bin on the same device
        CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);

        // Look for freed blocks from the active stream or from other idle streams
        bool found = false;
        while ( (block_itr != cached_blocks.end())
                && (block_itr->device == device)
                && (block_itr->bin == search_key.bin)) {

          // use special rule for the last ("exact size") bin: set max memory overuse to 1/8th
          if (search_key.bin == (unsigned int) -1 && (block_itr->bytes - search_key.bytes)*8UL > search_key.bytes)
            break;

          cudaStream_t prev_stream = block_itr->associated_stream;
	  if ((active_stream == prev_stream)
	      || (cudaEventQuery(block_itr->ready_event) != cudaErrorNotReady)) {
	    // Reuse existing cache block.  Insert into live blocks.
	    found = true;
	    search_key = *block_itr;
	    search_key.associated_stream = active_stream;
	    live_blocks.insert(search_key);

	    // Remove from free blocks
	    cached_blocks.erase(block_itr);
	    cached_bytes[device].free -= search_key.bytes;
	    cached_bytes[device].busy += search_key.bytes;

	    if (debug) CubLog("\tdevice %d reused cached block at %p (%lld bytes) for stream %lld (previously associated with stream %lld).\n",
			      device, search_key.d_ptr, (long long) search_key.bytes, (long long) search_key.associated_stream, (long long)  prev_stream);

	    break;
	  }

	  block_itr++;
	}
	// done searching. Unlock.

	Unlock(&spin_lock);

	if (!found)
	  {

	    // Set to specified device. Entrypoint may not be set.
	    if (device != entrypoint_device) {
	      if (CubDebug(error = cudaGetDevice(&entrypoint_device)))
		return error;
	      if (CubDebug(error = cudaSetDevice(device))) return error;
	    }

	    // Allocate
	    error = cudaMalloc(&search_key.d_ptr, search_key.bytes);

	    if (error != cudaSuccess) {
	      if (debug) CubLog("\tdevice %d failed to allocate %lld bytes for stream %lld",
				device, (long long) search_key.bytes, (long long) search_key.associated_stream);
	    }
	    if (CubDebug(error))
	      return error;
	    if (CubDebug(error = cudaEventCreateWithFlags(&search_key.ready_event, cudaEventDisableTiming)))
	      return error;

	    // Insert into live blocks
	    Lock(&spin_lock);
	    live_blocks.insert(search_key);
	    cached_bytes[device].busy += search_key.bytes;
	    Unlock(&spin_lock);

	    if (debug) CubLog("\tdevice %d allocated new device block at %p (%lld bytes associated with stream %lld).\n",
			      device, search_key.d_ptr, (long long) search_key.bytes, (long long) search_key.associated_stream);

	    // Attempt to revert back to previous device if necessary
	    if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device))
	      {
		if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
	      }
	  }

	// Copy device pointer to output parameter
	*d_ptr = search_key.d_ptr;
	if (debug) CubLog("\t\t%lld available blocks cached (%lld bytes), %lld live blocks outstanding(%lld bytes).\n",
			  (long long) cached_blocks.size(), (long long) cached_bytes[device].free, (long long) live_blocks.size(), (long long) cached_bytes[device].busy);

        return error;

    #endif // CUB_PTX_ARCH
    }


    /**
     * \brief Provides a suitable allocation of device memory for the given size on the current device.
     *
     * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
     * with which it was associated with during allocation, and it becomes available for reuse within other
     * streams when all prior work submitted to \p active_stream has completed.
     */
    cudaError_t DeviceAllocate(
        void            **d_ptr,            ///< [out] Reference to pointer to the allocation
        size_t          bytes,              ///< [in] Minimum number of bytes for the allocation
        cudaStream_t    active_stream = 0)  ///< [in] The stream to be associated with this allocation
    {
    #if (CUB_PTX_ARCH > 0)
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else
        return DeviceAllocate(INVALID_DEVICE_ORDINAL, d_ptr, bytes, active_stream);
    #endif // CUB_PTX_ARCH
    }


    /**
     * \brief Frees a live allocation of device memory on the specified device, returning it to the allocator.
     *
     * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
     * with which it was associated with during allocation, and it becomes available for reuse within other
     * streams when all prior work submitted to \p active_stream has completed.
     */
    cudaError_t DeviceFree(
        int             device,
        void*           d_ptr)
    {
    #if (CUB_PTX_ARCH > 0)
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        int entrypoint_device           = INVALID_DEVICE_ORDINAL;
        cudaError_t error               = cudaSuccess;

        if (device == INVALID_DEVICE_ORDINAL) {
	 if (CubDebug(error = cudaGetDevice(&entrypoint_device)))
	   return error;
         device = entrypoint_device;
	}

        BlockDescriptor search_key(d_ptr, device);
	bool recached = false;

	// Lock
	Lock(&spin_lock);

	// Find corresponding block descriptor
	BusyBlocks::iterator block_itr = live_blocks.find(search_key);
	if (block_itr != live_blocks.end()) {
	  // Remove from live blocks
	  search_key = *block_itr;
	  live_blocks.erase(block_itr);
	  cached_bytes[device].busy -= search_key.bytes;

	  // Check if we should keep the returned allocation
	  if (cached_bytes[device].free + search_key.bytes <= max_cached_bytes)
	    {
	      // Insert returned allocation into free blocks
	      cached_blocks.insert(search_key);
	      cached_bytes[device].free += search_key.bytes;
	      recached = true;
	      if (debug) {
		CubLog("\tdevice %d returned %lld bytes from associated stream %lld.\n\t\t %lld available blocks cached (%lld bytes), %lld live blocks outstanding. (%lld bytes)\n",
		       device, (long long) search_key.bytes, (long long) search_key.associated_stream, (long long) cached_blocks.size(),
		       (long long) cached_bytes[device].free, (long long) live_blocks.size(), (long long) cached_bytes[device].busy);
	      }
	    }
	}

        Unlock(&spin_lock);

        if (recached) {
	      // Signal the event in the associated stream
	      if (CubDebug(error = cudaEventRecord(search_key.ready_event, search_key.associated_stream)))
		return error;
	} else  {
	  // Set to specified device. Entrypoint may not be set.
	  if (device != entrypoint_device) {
	    if (CubDebug(error = cudaGetDevice(&entrypoint_device)))
	      return error;
	    if (CubDebug(error = cudaSetDevice(device))) return error;
	  }

          // Actually free device memory
          if (CubDebug(error = cudaFree(d_ptr))) return error;
          if (CubDebug(error = cudaEventDestroy(search_key.ready_event))) return error;

          if (debug) CubLog("\tdevice %d freed %lld bytes from associated stream %lld.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
                            device, (long long) search_key.bytes, (long long) search_key.associated_stream, (long long) cached_blocks.size(), (long long) cached_bytes[device].free, (long long) live_blocks.size(), (long long) cached_bytes[device].busy);

	  if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device))
	    {
	      if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
	    }
        }

        return error;

    #endif // CUB_PTX_ARCH
    }


    /**
     * \brief Frees a live allocation of device memory on the current device, returning it to the allocator.
     *
     * Once freed, the allocation becomes available immediately for reuse within the \p active_stream
     * with which it was associated with during allocation, and it becomes available for reuse within other
     * streams when all prior work submitted to \p active_stream has completed.
     */
    cudaError_t DeviceFree(
        void*           d_ptr)
    {
    #if (CUB_PTX_ARCH > 0)
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else
        return DeviceFree(INVALID_DEVICE_ORDINAL, d_ptr);
    #endif // CUB_PTX_ARCH
    }


    /**
     * \brief Frees all cached device allocations on all devices
     */
    cudaError_t FreeAllCached()
    {
    #if (CUB_PTX_ARCH > 0)
        // Caching functionality only defined on host
        return CubDebug(cudaErrorInvalidConfiguration);
    #else

        cudaError_t error         = cudaSuccess;
        int entrypoint_device     = INVALID_DEVICE_ORDINAL;
        int current_device        = INVALID_DEVICE_ORDINAL;

        Lock(&spin_lock);

        while (!cached_blocks.empty())
        {
            // Get first block
            CachedBlocks::iterator begin = cached_blocks.begin();

            // Get entry-point device ordinal if necessary
            if (entrypoint_device == INVALID_DEVICE_ORDINAL)
            {
                if (CubDebug(error = cudaGetDevice(&entrypoint_device))) break;
            }

            // Set current device ordinal if necessary
            if (begin->device != current_device)
            {
                if (CubDebug(error = cudaSetDevice(begin->device))) break;
                current_device = begin->device;
            }

            // Free device memory
            if (CubDebug(error = cudaFree(begin->d_ptr))) break;
            if (CubDebug(error = cudaEventDestroy(begin->ready_event))) break;

            // Reduce balance and erase entry
            cached_bytes[current_device].free -= begin->bytes;
            cached_blocks.erase(begin);

            if (debug) CubLog("\tdevice %d freed %lld bytes.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
                              current_device, (long long) begin->bytes, (long long) cached_blocks.size(), (long long) cached_bytes[current_device].free, (long long) live_blocks.size(), (long long) cached_bytes[current_device].busy);
        }

        Unlock(&spin_lock);

        // Attempt to revert back to entry-point device if necessary
        if (entrypoint_device != INVALID_DEVICE_ORDINAL)
        {
            if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
        }

        return error;

    #endif // CUB_PTX_ARCH
    }


    /**
     * \brief Destructor
     */
    virtual ~CachingDeviceAllocator()
    {
        if (!skip_cleanup)
            FreeAllCached();
    }

};




/** @} */       // end group UtilMgmt

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
