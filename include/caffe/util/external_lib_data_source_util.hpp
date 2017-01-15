/**
* @brief    This file contains helper interface declarations for external
*           library data source.
*/
#ifndef EXTERNAL_LIB_DATA_SOURCE_UTIL_H_
#define EXTERNAL_LIB_DATA_SOURCE_UTIL_H_

#include <boost/shared_ptr.hpp>
#include <string>

class IExternalLibDataSource;

/**
* @brief    Class that abstracts away platform dependent library handling.
*/
class IExternalLib {
 public:
  virtual ~IExternalLib() {}
  virtual IExternalLibDataSource* GetDataSource() = 0;
};

/**
* @brief   Returns library object that abstracts away library management
*          dependent on platform.
*/
boost::shared_ptr<IExternalLib> GetDataSourceLibraryWrapper(
  const std::string& ext_lib_path, const std::string& factory_name,
  const std::string& ext_lib_param);

#endif
