#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <opencv2/core/core.hpp>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param);
  virtual ~DataTransformer() {}

  void InitRand();

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param input_blob
   *    A Blob containing the data to be transformed.
   * @param transformed_data
   *    This should point to the right place in memory where the 
   *    transformed data would be written. Assumes there is enough space
   * @param transformed_blob
   *    This is destionation blob. It can be part of top blob's data. 
   *    The transformed data will be  written at the appropriate place
   *    within the blob's data.
   */

  void Transform(const Datum& datum, Dtype* transformed_data);

  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);

  void Transform(const vector<Datum> & datum_vector,
                Blob<Dtype>* transformed_blob);

  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);

  void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

 protected:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   * 
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return 
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);

  // Tranformation parameters
  TransformationParameter param_;


  shared_ptr<Caffe::RNG> rng_;
  Caffe::Phase phase_;
  Blob<Dtype> data_mean_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_

