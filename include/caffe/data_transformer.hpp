#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#ifndef OSX
#include <opencv2/core/core.hpp>
#endif

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

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  /**
   * @brief Reset Transformation state. This would cause to generate random params
   *    the next Transform is called
   */
  void ResetState();

  /**
   * @brief Tells if the Transformation is persistent or not
   */
  inline bool IsPersistent() { return state_.persistent; }

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used See data_layer.cpp for an example.
   */
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used See memory_layer.cpp for an example.
   */
  void Transform(const vector<Datum> & datum_vector,
                Blob<Dtype>* transformed_blob);

#ifndef OSX
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);
    /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of cv::Mat.
   *
   * @param cv_img_vector
   *    A vector of cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used See memory_layer.cpp for an example.
   */
  void Transform(const vector<cv::Mat>& cv_img_vector,
                Blob<Dtype>* transformed_blob);
#endif

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to all the num images in a input_blob.
   *
   * @param input_blob
   *    A Blob containing the data to be transformed. It applies a
   *    transformation to each the num images in the blob.
   * @param transformed_blob
   *    This is destination blob, it will contain as many images as the
   *    input blob. It can be part of top blob's data.
   */
  void Transform(const Blob<Dtype>& input_blob, Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to all the Blobs in input_blobs.
   *
   * @param input_blobs
   *    A Vector of Blob containing the data to be transformed. 
   *    It applies the transformation to each of the Blobs int Vector.
   * @param transformed_blobs
   *    This is destination vector of blob, it will contain as many 
   *    Blobs input_blobs.
   */
  void Transform(const vector<Blob<Dtype>*>& input_blobs,
                const vector<Blob<Dtype>*>& transformed_blobs);

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

  struct TransformState {
    bool persistent;
    bool reset;
    bool do_mirror;
    int h_off;
    int w_off;
    int input_channels;
    int input_height;
    int input_width;
  };

  void UpdateState(const int input_channels, const int input_height,
                  const int input_width);

  void CheckSizes(const int input_channels, const int input_height,
    const int input_width, const int output_channels,
    const int output_height, const int output_width);

  // Do the actual transformation for different Datatypes:
  // uchar, uint8_t, float, double
  // For Datum or Blob (continous) use:
  //  num_blocks = 1
  //  height_offset = input_width
  //  channel_offset = input_height * input_width;
  // For cv::Mat (continous) use:
  //  num_blocks = 1
  //  height_offset = input_width
  //  channel_offset = 1;
  // For cv::Mat (discontinous) use:
  //  num_blocks = input_height
  //  height_offset = 0
  //  channel_offset = 1;
  void InternalTransform(const vector<const uchar*> & data_ptrs,
    const int height_offset, const int channel_offset,
    const int output_height, const int output_width,
    const int output_channels, Dtype* transformed_data);

  void InternalTransform(const vector<const char*> & data_ptrs,
    const int height_offset, const int channel_offset,
    const int output_height, const int output_width,
    const int output_channels, Dtype* transformed_data);

  void InternalTransform(const vector<const float*> & data_ptrs,
    const int height_offset, const int channel_offset,
    const int output_height, const int output_width,
    const int output_channels, Dtype* transformed_data);

  void InternalTransform(const vector<const double*> & data_ptrs,
    const int height_offset, const int channel_offset,
    const int output_height, const int output_width,
    const int output_channels, Dtype* transformed_data);

  // Tranformation parameters
  TransformationParameter param_;

  shared_ptr<Caffe::RNG> rng_;
  Caffe::Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  TransformState state_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_

