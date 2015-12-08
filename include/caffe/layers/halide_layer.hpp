#ifndef CAFFE_LAYERS_HALIDE_HPP_
#define CAFFE_LAYERS_HALIDE_HPP_

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


//typedef void (*func_t)(buffer_t *bottom_buf_, buffer_t *top_buff_);


template <typename Dtype>
class HalideLayer : public Layer<Dtype> {
 public:
  explicit HalideLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        h_Forward_cpu(NULL),
        h_Backward_cpu(NULL),
        h_Forward_gpu(NULL),
        h_Backward_gpu(NULL)
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

 private:
  // Creator and destructors of external instance
  /*
  create_t* create_ext;
  destroy_t* destroy_ext;
  HalideExtern* inst_ext;
  */

  //void (*extern_Forward_gpu)(buffer_t *bottom_buf_, buffer_t *top_buff_);

  typedef void (*hfunc_t)(buffer_t *bottom_buf_, buffer_t *top_buff_);
  hfunc_t h_Forward_cpu;
  hfunc_t h_Backward_cpu;
  hfunc_t h_Forward_gpu;
  hfunc_t h_Backward_gpu;


  buffer_t bottom_buf_;
  buffer_t bottom_diff_buf_;
  buffer_t p_buf_;
  buffer_t p_diff_buf_;
  buffer_t top_buf_;
  buffer_t top_diff_buf_;
  int stride_;

  void* halide_lib_handle;

};

}  // namespace caffe

#endif
