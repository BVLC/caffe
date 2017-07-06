#ifndef CAFFE_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#ifdef USE_GREENTEA
#include <boost/filesystem.hpp>
#endif

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

enum gemm_type_t {
  GEMM_TYPE_DEFAULT = 0,
  GEMM_TYPE_FAST_IMAGE_32_1,
  GEMM_TYPE_FAST_IMAGE_32_2,
  GEMM_TYPE_FAST_IMAGE_B_IMAGE,
  GEMM_TYPE_FAST_BUFFER
};
template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
#ifdef USE_GREENTEA
    weight_image_ = NULL;
    weight_image_seq_ = -1;
    innerprod_type_ = GEMM_TYPE_DEFAULT;
    tuned_ = false;

    if (std::getenv("CLCAFFE_CACHE_PATH"))
      cache_path_ << std::getenv("CLCAFFE_CACHE_PATH");
    else if (std::getenv("VIENNACL_CACHE_PATH"))
      cache_path_ << std::getenv("VIENNACL_CACHE_PATH") << "/clCaffe";
    else if (std::getenv("HOME")) {
      cache_path_ << std::getenv("HOME") << "/.cache/clCaffe";
    }
    cache_path_ << "/innerprod/";
    const boost::filesystem::path& path = cache_path_.str();
    const boost::filesystem::path& dir =
                   boost::filesystem::unique_path(path).string();
    bool hasCacheDir = false;
    if (!boost::filesystem::exists(dir))
      hasCacheDir = boost::filesystem::create_directories(dir);
    else
      hasCacheDir = boost::filesystem::is_directory(dir);

    if (hasCacheDir != true) {
      std::cout << "Failed to create cache directory,"
                << "will tune again for next running" << std::endl;
    }
#endif
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InnerProduct"; }
  virtual inline int_tp ExactNumBottomBlobs() const { return 1; }
  virtual inline int_tp ExactNumTopBlobs() const { return 1; }
#ifdef USE_GREENTEA
  ~InnerProductLayer() {
    if (weight_image_)
      clReleaseMemObject(weight_image_);
    weight_image_ = NULL;
  }
#endif

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
#ifdef USE_GREENTEA
  virtual void generate_key();
  virtual void tune_innerprod_type(const int_tp ctx_id,
       const CBLAS_TRANSPOSE TransB, const cl_mem A,
       const cl_mem B, const cl_mem B_image, const size_t max_image_size);
  virtual bool load_cache();
#endif

  int_tp M_;
  int_tp K_;
  int_tp N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights
#ifdef USE_GREENTEA
  cl_mem weight_image_;
  const SyncedMemory * copied_weight_data_;
  bool test_only_;
  uint64_t weight_image_seq_;
  gemm_type_t innerprod_type_;
  bool tuned_;
  std::stringstream cache_path_;
  std::string key_;
#endif
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
