/**
* @brief    This file contains interface declarations that external data source
*           needs to implement to be able to plug into caffe pipeline. External
*           library needs to export the factory method with the signature
*           defined below.
*           To instruct caffe to use external lib as data source one needs to
*           use external lib data layer.
*/
#ifndef EXTARNAL_LIB_DATA_SOURCE_H_
#define EXTARNAL_LIB_DATA_SOURCE_H_

class IExternalLibDataSource;

// External library data source factory method signature.
typedef IExternalLibDataSource* (*ExternalLibDataSourceFactoryMethod)(
  const char* external_lib_params);

enum BlobType { BlobTypeFLOAT, BlobTypeDOUBLE };

/**
* @brief    Defines interface for one data point.
*/
class IDatum {
 public:
  // Returns datum shape for the blob with the given name.
  virtual void GetBlobShape(const char* blob_name, const int** shape,
    int* shape_count) = 0;

  // Returns data of the given type for the blob with the given name.
  virtual void GetBlobData(const char* blob_name, void* blob_data,
    BlobType type) = 0;
};

/**
* @brief    Defines interface for data source.
*/
class IExternalLibDataSource {
 public:
  // Releases resources, data source should not be used after this call.
  virtual void Release() = 0;

  // Moves internal iterator to the next datum.
  virtual void MoveToNext() = 0;

  // Returns current datum. Returned object must be valid till the next
  // MoveToNext call. This call does not advances internal iterator,
  // MoveToNext needs to be called to advance.
  virtual IDatum* GetCurrent() = 0;
};

#endif
