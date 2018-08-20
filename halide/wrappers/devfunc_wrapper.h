#include <string>
#include <vector>

#include "../sync_halide.hpp"

#include "caffe/caffe.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

using namespace std;
using namespace caffe;


template< typename Dtype >
class DevFuncWrapper : public Layer<Dtype> {
public:
  explicit DevFuncWrapper(const LayerParameter& param)
      : Layer<Dtype>(param),
        param_(param.python_param()){}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs();
  virtual inline int ExactNumTopBlobs();

  protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

private:
  PythonParameter param_;

  std::vector< std::string > bottom_blob_names;
  std::vector< std::string > top_blob_names;

  std::vector< std::pair<int, int> > caffe_from_halide_;
  buffer_t bottom_buf_;
  buffer_t bottom_diff_buf_;
  buffer_t top_buf_;
  buffer_t top_diff_buf_;

};

template class DevFuncWrapper<float>;
template class DevFuncWrapper<double>;
