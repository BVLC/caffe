#ifndef CAFFE_DOWNLOAD_MANAGER_H_
#define CAFFE_DOWNLOAD_MANAGER_H_

#include <curlpp/cURLpp.hpp>
#include <curlpp/Multi.hpp>

#include <sstream>
#include <string>
#include <vector>

#include "caffe/common.hpp"

namespace caffe {

using std::string;
using std::stringstream;

class DownloadManager {
 public:
  virtual ~DownloadManager() { }
  void AddUrl(const string& url);
  const vector<shared_ptr<stringstream> >& RetrieveResults() const;
  void Reset();
  virtual void Download();

  static DownloadManager* DefaultDownloadManagerFactory() {
    return new DownloadManager();
  }

 protected:
  void Register(const string& url);

  vector<shared_ptr<stringstream> > streams_;
  vector<string> urls_;
  curlpp::Multi curl_multi_;
};

}  // namespace caffe

#endif  // CAFFE_DOWNLOAD_MANAGER_H_
