#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::min;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
        "The backend {leveldb, lmdb} containing the images");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Compute the channel-wise mean and principal "
        " components of a set of images given by a leveldb/lmdb\n"
        "Usage:\n"
        "    compute_image_pca [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 2 || argc > 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_pca");
    return 1;
  }

  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[1], db::READ);
  scoped_ptr<db::Cursor> cursor(db->NewCursor());

  Datum first_datum;
  first_datum.ParseFromString(cursor->value());

  if (DecodeDatumNative(&first_datum)) {
    LOG(INFO) << "Decoding Datum";
  }

  const int data_size = first_datum.channels() * first_datum.height()
    * first_datum.width();

  int size_in_datum = std::max<int>(first_datum.data().size(),
                                    first_datum.float_data_size());
  const int channels = first_datum.channels();
  const int dim = first_datum.height() * first_datum.width();

  cv::Mat covar = cv::Mat::zeros(channels, channels, CV_64F);
  cv::Mat covar_prev = cv::Mat::zeros(channels, channels, CV_64F);
  cv::Mat mean = cv::Mat::zeros(1, channels, CV_64F);
  cv::Mat mean_prev = cv::Mat::zeros(1, channels, CV_64F);

  int count = 0;

  // We calculate both mean and covariance online, since for large datasets
  // i.e. Imagenet, we can easily overflow using the naive algorithm.
  // see
  // http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  LOG(INFO) << "Calculating mean and principal components...";
  while (cursor->valid()) {
    Datum datum;
    datum.ParseFromString(cursor->value());
    DecodeDatumNative(&datum);

    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;

    const std::string& data = datum.data();
    if (data.size() != 0) {
      uint64_t n_pixels = 0;
      CHECK_EQ(data.size(), size_in_datum);
      for (int i = 0; i < dim; ++i) {
        n_pixels = count*dim + (i+1);

        // for each pixel, update mean and covariance for all channels
        for (int c = 0; c < channels; ++c) {
          mean.ptr<double>(0)[c] +=
              (((uint8_t) data[dim * c + i])
                  - mean.ptr<double>(0)[c])/n_pixels;
        }
        for (int c_i = 0; c_i < channels; c_i++) {
          for (int c_j = 0; c_j <= c_i; c_j++) {
            covar.ptr<double>(c_i)[c_j] =
                (covar_prev.ptr<double>(c_i)[c_j]*(n_pixels-1)
                + (((uint8_t) data[dim * c_i + i])
                    - mean.ptr<double>(0)[c_i] )
                * (((uint8_t) data[dim * c_j + i])
                    - mean_prev.ptr<double>(0)[c_j]))/n_pixels;
          }
        }

        covar.copyTo(covar_prev);
        mean.copyTo(mean_prev);
      }
    } else {
      uint64_t n_pixels = 0;
      CHECK_EQ(datum.float_data_size(), size_in_datum);
      for (int i = 0; i < dim; ++i) {
        n_pixels = count*dim + (i+1);

        // for each pixel, update mean and covariance for all channels
        for (int c = 0; c < channels; ++c) {
          mean.ptr<double>(0)[c] +=
              (static_cast<float>(datum.float_data(dim * c + i))
                  - mean.ptr<double>(0)[c])/n_pixels;
        }
        for (int c_i = 0; c_i < channels; c_i++) {
          for (int c_j = 0; c_j <= c_i; c_j++) {
            covar.ptr<double>(c_i)[c_j] =
                (covar_prev.ptr<double>(c_i)[c_j]*(n_pixels-1)
                    + (static_cast<float>(datum.float_data(dim * c_i + i))
                        - mean.ptr<double>(0)[c_i])
                    * (static_cast<float>(datum.float_data(dim * c_j + i))
                        - mean_prev.ptr<double>(0)[c_j]))/n_pixels;
          }
        }

        covar.copyTo(covar_prev);
        mean.copyTo(mean_prev);
      }
    }

    ++count;
    if (count % 1000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
      LOG(INFO) << "Mean" << mean;
      LOG(INFO) << "Sample Covariance" << covar;
    }
    cursor->Next();
  }

  // fill in uncalculated symmetric part of matrix
  for (int c_i = 0; c_i < channels; c_i++) {
    for (int c_j = 0; c_j < c_i; c_j++) {
      covar.ptr<double>(c_j)[c_i] = covar.ptr<double>(c_i)[c_j];
    }
  }

  LOG(INFO) << "Processed " << count << " files.";
  LOG(INFO) << "Mean channel values: " << mean;
  LOG(INFO) << "Channel Covariance: " << covar;

  cv::Mat eigenvalues, eigenvectors;
  cv::eigen(covar, eigenvalues, eigenvectors);

  for (int c = 0; c < channels; ++c) {
    LOG(INFO) << "mean_value: " << mean.ptr<double>(0)[c];
  }
  for (int c = 0; c < channels; ++c) {
    LOG(INFO) << "eigen_value: " << eigenvalues.ptr<double>(0)[c];
  }
  for (int i = 0; i < channels; ++i) {
    for (int j = 0; j < channels; ++j) {
      LOG(INFO) << "eigen_vector_component: " << eigenvectors.ptr<double>(i)[j];
    }
  }

  return 0;
}
