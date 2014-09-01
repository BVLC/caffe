#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>
#include <vector>

#include "../python/caffe/_caffe.hpp"
#include "caffe/layer.hpp"

namespace caffe {

/**
 * @brief Wrap a layer implemented in Python.
 */
template <typename Dtype>
class PythonLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides python_param, with required parameters:
   * - module. The module to import with the layer implementation. Note that
   *   the current directory is not in the module search path by default.
   * - layer. The name of the layer class, which must implement setup
   *   (for LayerSetUp), reshape (for Reshape), forward (for Forward_cpu), and
   *   backward (for Backward_cpu).
   */
  explicit PythonLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_PYTHON;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  boost::python::object layer_;

 private:
  vector<PyBlob<Dtype> > PythonBlobVector(const vector<Blob<Dtype>*>& vec);
};

}  // namespace caffe

#endif
