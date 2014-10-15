#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>

#include <string>
#include <vector>

#include "caffe/util/download_manager.hpp"

#include "caffe/util/benchmark.hpp"

namespace caffe {

void DownloadManager::AddUrl(const string& url) {
  DLOG(INFO) << url;

  urls_.push_back(url);
}

void DownloadManager::Register(const string& url) {
  DLOG(INFO) << "registering " << url;

  curlpp::Easy *e = new curlpp::Easy;

  stringstream* stream = new stringstream();
  streams_.push_back(shared_ptr<stringstream>(stream));

  e->setOpt(new curlpp::options::WriteStream(stream));
  e->setOpt(new curlpp::Options::FollowLocation(true));
  e->setOpt(new curlpp::Options::Url(url.c_str()));
  e->setOpt(new curlpp::Options::Timeout(200));

  curl_multi_.add(e);
}

void DownloadManager::Download() {
  // TODO(kmatzen): Do I need this cleanup?  Read the cURLpp docs.
  // curlpp::Cleanup cleanup;

  const int kMaxHandles = 32;
  int running_handles = -1;
  fd_set read, write, exc;
  int max_fd = -1;
  int64_t timeout = -1;
  timeval sel_timeout;

  vector<string>::const_iterator url_iter = urls_.begin() + streams_.size();

  DLOG(INFO) << "preparing requests...";
  // TODO(kmatzen): I sort of manually enforce this, but can I tell cURLpp to
  // do it for me?
  // curl_multi_.setOpt(new curlpp::Options::MaxConnections(kMaxHandles));

  for (int i = 0; i < kMaxHandles && url_iter != urls_.end(); ++url_iter, ++i) {
    Register(*url_iter);
  }

  DLOG(INFO) << "BEGIN DOWNLOAD";
  while (running_handles) {
    while (!curl_multi_.perform(&running_handles)) {
    }

    if (!running_handles) {
      break;
    }

    FD_ZERO(&read);
    FD_ZERO(&write);
    FD_ZERO(&exc);

    curl_multi_.fdset(&read, &write, &exc, &max_fd);
    // TODO(kmatzen): Figure out if curl needs to be configured with a timeout.
    // Sometimes it seems to get stuck and I'd like to just skip those examples.
    // curl_multi_.timeout(&timeout);

    if (timeout == -1) {
      timeout = 100;
    }

    if (max_fd == -1) {
      usleep(timeout * 1000);
    } else {
      sel_timeout.tv_sec = timeout / 1000;
      sel_timeout.tv_usec = (timeout % 1000) * 1000;

      if (select(max_fd + 1, &read, &write, &exc, &sel_timeout) < 0) {
        LOG(FATAL) << "select failed";
      }
    }

    const curlpp::Multi::Msgs msgs = curl_multi_.info();
    for (curlpp::Multi::Msgs::const_iterator msg = msgs.begin();
         msg != msgs.end(); ++msg) {
      if (msg->second.msg != CURLMSG_DONE) {
        continue;
      }

      const curlpp::Easy* easy_handle = msg->first;

      if (msg->second.code != CURLE_OK) {
        curlpp::Options::Url option;
        easy_handle->getOpt(option);
        LOG(ERROR) << "curl failed " << msg->second.code << " " << option;
      }

      curl_multi_.remove(easy_handle);
      delete easy_handle;

      if (url_iter == urls_.end()) {
        continue;
      }

      Register(*url_iter++);
    }
  }
  DLOG(INFO) << running_handles;
  DLOG(INFO) << "END DOWNLOAD";
}

const vector<shared_ptr<stringstream> >&
    DownloadManager::RetrieveResults() const {
  CHECK_EQ(urls_.size(), streams_.size());
  return streams_;
}

void DownloadManager::Reset() {
  streams_.clear();
  urls_.clear();
}

}  // namespace caffe
