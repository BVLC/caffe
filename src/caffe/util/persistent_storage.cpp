#include <boost/filesystem.hpp>

#include "caffe/util/persistent_storage.hpp"
#include "caffe/common.hpp"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <initguid.h>
#include <KnownFolders.h>
#include <ShlObj.h>
#include <wchar.h>
#include <comdef.h>
#endif  // _WIN32 || _WIN64

#ifdef defined(__APPLE__) || defined(__MACH__)
#include <CoreServices/CoreServices.h>
#endif  // __APPLE__ || __MACH__

namespace caffe {

string persistent_storage_path() {
  string path = "";

#if defined(_WIN32) || defined(_WIN64)
  LPWSTR wszPath = NULL;
  HRESULT hr = SHGetKnownFolderPath(FOLDERID_RoamingAppData,
                                    KF_FLAG_CREATE, NULL, &wszPath);
  _bstr_t bstrPath(wszPath);
  string tmp((char*)bstrPath);
  CoTaskMemFree(wszPath);
  path = tmp + "/caffe";
#endif  // _WIN32 || _WIN64


#if defined(__linux__) || defined(__linux)
  char* path_c = getenv("XDG_DATA_HOME");
  if (path_c) {
    string tmp(path_c);
    path = tmp + "/caffe";
  } else {
    path_c = getenv("HOME");
    string tmp(path_c);
    path = tmp + "/.local/share/caffe";
  }
#endif  // __linux__ || __linux


#if defined(__APPLE__) || defined(__MACH__)
  FSRef ref;
  OSType folderType = kApplicationSupportFolderType;
  char path_c[PATH_MAX];
  FSFindFolder(kUserDomain, folderType, kCreateFolder, &ref);
  FSRefMakePath( &ref, (UInt8*)&path_c, PATH_MAX);
  string tmp(path_c);
  path = tmp + "/caffe";
#endif  // __APPLE__ || __MACH__

#ifdef CAFFE_STORAGE_PATH_OVERRIDE
  string override_path = CAFFE_STORAGE_PATH_OVERRIDE;
  if (override_path.size() > 0) {
    path = override_path;
  }
#endif  // CAFFE_STORAGE_PATH_OVERRIDE

#ifndef NDEBUG
  std::cout << "Persistent storage path: " << path << std::endl;
#endif  // NDEBUG

  boost::filesystem::path dir(path);
  boost::filesystem::create_directories(dir);

  return path + "/";
}

}  // namespace caffe
