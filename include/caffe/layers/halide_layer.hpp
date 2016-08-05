#ifndef CAFFE_LAYERS_HALIDE_HPP_
#define CAFFE_LAYERS_HALIDE_HPP_

#include <dlfcn.h>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/halide.hpp"



namespace caffe {

template< typename Dtype>
class DLLInterface {
 public:
  virtual ~DLLInterface() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;

  virtual inline int ExactNumBottomBlobs() = 0;
  virtual inline int ExactNumTopBlobs() = 0;

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;
};
template class DLLInterface<float>;
template class DLLInterface<double>;




template <typename Dtype>
class HalideLayer : public Layer<Dtype> {
 public:
  explicit HalideLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        param_(param.halide_param()),
        create_ext(NULL),
        destroy_ext(NULL),
        inst_ext(NULL) {}
  virtual ~HalideLayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "Halide"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  HalideParameter param_;


 private:
  void* halide_lib_handle;


  // the types of the class factories
  typedef DLLInterface<Dtype>* create_t();
  typedef void destroy_t(DLLInterface<Dtype>*);

  // Creator and destructors of external instance
  create_t* create_ext;
  destroy_t* destroy_ext;
  DLLInterface<Dtype>* inst_ext;

  buffer_t bottom_buf_;
  buffer_t bottom_diff_buf_;
  buffer_t top_buf_;
  buffer_t top_diff_buf_;
};

}  // namespace caffe

#endif
