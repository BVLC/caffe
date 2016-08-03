#ifndef CAFFE_LAYERS_HALIDE_HPP_
#define CAFFE_LAYERS_HALIDE_HPP_

#include <dlfcn.h>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/halide_layer.hpp"
#include "caffe/util/halide.hpp"



namespace caffe {

/*
class ExternClass {
protected:
    double side_length_;
public:
    HalideExtern() {}
    ~HalideExtern() {}

    //void set_side_length(double side_length) {
    //    side_length_ = side_length;
    //}
    virtual void Forward_gpu();
    virtual void Backward_gpu();
};
// the types of the class factories
typedef HalideExtern* create_t();
typedef void destroy_t(HalideExtern*);
*/

template <typename Dtype>
class HalideLayer : public Layer<Dtype> {
 public:
  explicit HalideLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        param_(param.halide_param()),
        hv_Forward_cpu(NULL),
        hv_Backward_cpu(NULL),
        hv_Forward_gpu(NULL),
        hv_Backward_gpu(NULL)
      /*,
        create_ext(NULL),
        destroy_ext(NULL),
        inst_ext(NULL)
     */
     {}
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

  std::vector< std::pair<int, int> > caffe_from_halide_;

 private:
  // Creator and destructors of external instance
  /*
  create_t* create_ext;
  destroy_t* destroy_ext;
  HalideExtern* inst_ext;
  */

  typedef void (*hvfunc_t)(void* argv);

  hvfunc_t hv_Forward_cpu;
  hvfunc_t hv_Backward_cpu;
  hvfunc_t hv_Forward_gpu;
  hvfunc_t hv_Backward_gpu;

  buffer_t bottom_buf_;
  buffer_t bottom_diff_buf_;
  buffer_t top_buf_;
  buffer_t top_diff_buf_;

  void* halide_lib_handle;
};

}  // namespace caffe

#endif
