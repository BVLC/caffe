#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/layers/halide_layer.hpp"

using namespace std;
using namespace caffe;


template< typename Dtype >
class PlipWrapper : public DLLInterface<Dtype> {
public:
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
  std::vector< std::string > bottom_blob_names;
  std::vector< std::string > top_blob_names;

  std::vector< std::pair<int, int> > caffe_from_halide_;
  buffer_t bottom_buf_;
  buffer_t bottom_diff_buf_;
  buffer_t top_buf_;
  buffer_t top_diff_buf_;

};

// template class specializations
template class PlipWrapper<float>;
template class PlipWrapper<double>;
