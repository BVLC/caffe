#include "caffe/common.hpp"

#define modHash(n) ((n)%(2*table_capacity));

namespace caffe {
  template<int kd>
  __device__ __host__ static unsigned int hash(signed short *key) {
    unsigned int k = 0;
    for (int i = 0; i < kd; i++) {
      k += key[i];
      k  = k * 1664525;
    }
    return k;
  }
  template<int kd>
  __device__ __host__ static unsigned int has(int *key) {
    unsigned int k = 0;
    for (int i = 0; i < kd; i++) {
      k += key[i];
      k  = k * 1664525;
    }
    return k;
  }
  template<int kd>
  __device__ static int hashTableInsert(unsigned int fh, signed short *key,
    signed short * table_keys,
    int* table_entries,
    int table_capacity,
    unsigned int slot)
    {
      int h = modHash(fh);
      while (1) {
        int *e = &table_entries[h];
        // if the cell is empty (-1), lock it (-2)
        int contents = atomicCAS(e, -1, -2);

        if (contents == -2) {
          // If it was locked already, move on the next cell

        } else if (contents == -1) {
          // If it was empty, we successfully locked it, write our key
          for (int i = 0; i < kd; i++) {
            table_keys[slot*kd+i] = key[i];
          }
          // Unlock
          atomicExch(e, slot);

          return h;
        } else {
          // The cell is unlocked and has a key in it, check if it matches
          bool match = true;
          for (int i = 0; i < kd && match; i++) {
            match  = (table_keys[contents*kd+i] == key[i]);
          }
          if (match) return h;
        }
        // increment the bucket with wraparound
        h++;
        if (h == table_capacity*2) h = 0;
      }
    }

    template<int kd>
    __device__ static int hashTableInsert(signed short *key,
    signed short* table_keys,
    int* table_entries,
    int table_capacity,
    unsigned int slot) {
      unsigned int myHash = hash<kd>(key);
      return hashTableInsert<kd>(myHash, key, table_keys, table_entries, table_capacity, slot);
    }

    template<int kd>
    __device__ static int hashTableRetrieve(signed short*key,
      const int * table_entries,
      const signed short* table_keys,
      const int table_capacity) {
        int h = modHash(hash<kd>(key));
        while (1) {
          const int *e = table_entries + h;
          if (*e == -1) return -1;
          bool match = true;
          for (int i = 0; i < kd && match; i++) {
            match = (table_keys[(*e)*kd+i] == key[i]);
          }
          if (match) return *e;

          h++;
          if (h == table_capacity*2) h = 0;
        }
      }

} //namespace caffe

