#ifndef CAFFE_HASH_TABLE_HPP
#define CAFFE_HASH_TABLE_HPP

#include "caffe/common.hpp"

//TODO : Might be better to unify this with already existing HashTable class
namespace caffe{

class HashTable
{
  public:
    int *table_entries;
    unsigned int table_capacity;
    signed short *table_keys;
    bool create;

    HashTable() : create(false) {}
    
    void createHashTable(const int capacity, const int kd){
      #ifndef CPU_ONLY
      // TODO? use symbol to go in constant memory instead
      // Initialize table_capacity
      table_capacity = (unsigned int)capacity ;
      
      // Initialize table_entries
      CUDA_CHECK(cudaMalloc((void **) &table_entries, 2*capacity*sizeof(int)));
      CUDA_CHECK(cudaMemset(table_entries, -1, 2*capacity*sizeof(int)));
      
      // Initialize table_keys
      CUDA_CHECK(cudaMalloc((void **) &table_keys, capacity*kd*sizeof(signed short)));
      CUDA_CHECK(cudaMemset(table_keys, 0, capacity*kd*sizeof(signed short)));    

      // Set create to true
      create = true;
      #endif // CPU_ONLY
    }
    
    void resetHashTable(const int capacity, const int kd){
      #ifndef CPU_ONLY
      // TODO? use symbol to go in constant memory instead
      // Initialize table_capacity
      table_capacity = (unsigned int)capacity ;
      
      // Reset table_entries
      CUDA_CHECK(cudaMemset(table_entries, -1, 2*capacity*sizeof(int)));
      
      // Resettable_keys
      CUDA_CHECK(cudaMemset(table_keys, 0, capacity*kd*sizeof(signed short)));    
      #endif // CPU_ONLY   
    }
    
    ~HashTable(){
      #ifndef CPU_ONLY
      if(create){
        // Free pointers allocated during 
        CUDA_CHECK(cudaFree(table_entries));
        CUDA_CHECK(cudaFree(table_keys));
        }
      #endif //CPU_ONLY
    }

};
}//namespace caffe
#endif //CAFFE_HASH_TABLE_HPP
