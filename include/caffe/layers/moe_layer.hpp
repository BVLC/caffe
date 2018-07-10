#ifndef CAFFE_MOE_LAYER_HPP_
#define CAFFE_MOE_LAYER_HPP_


#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Forward declare net
template<typename Dtype>
class Net;


template<typename Dtype, typename MItype, typename MOtype>
class MOELayer : public Layer<Dtype, MItype, MOtype> {
 public:
  explicit MOELayer(const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
                          const vector<Blob<MOtype>*>& top);
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
                       const vector<Blob<MOtype>*>& top);

  virtual inline const char* type() const { return "MOE"; }

  virtual vector<shared_ptr<QuantizerBase> > get_all_quantizers();

 protected:
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);

  int_tp parallel_nets_;
  shared_ptr<Net<float> > gating_net_;
  Blob<MOtype>* gating_;
  vector<vector<shared_ptr<Net<float> > > > expert_nets_;

 private:
  void GenerateProgram();
};

}  // namespace caffe


#endif  // CAFFE_LAYERS_MOE_LAYER_HPP_
