#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "backend/device_kernel.hpp"
#include "backend/device_program.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/definitions.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/quantizer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "caffe/quantizer_creator.hpp"

#include "caffe/backend/backend.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/backend/opencl/caffe_opencl.hpp"
#include "caffe/backend/cuda/caffe_cuda.hpp"

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class mutex; }

namespace caffe {

class LayerBase {
 public:
  virtual ~LayerBase() {

  }

  virtual void Reshape(const vector<BlobBase*>& bottom,
                       const vector<BlobBase*>& top) = 0;

  virtual void Forward(const vector<BlobBase*>& bottom,
                      const vector<BlobBase*>& top,
                      void* out_loss) = 0;

  virtual void Backward(const vector<BlobBase*>& top,
                       const vector<bool>& propagate_down,
                       const vector<BlobBase*>& bottom) = 0;

  virtual void loss(const int_tp top_index, void* out_loss) = 0;

  /**
   * @brief Returns the layer parameter.
   */
  const LayerParameter& layer_param() const {
    return layer_param_;
  }
  /**
   * @brief Returns the device context this layer runs on
   */
  inline Device *get_device() {
    return device_;
  }
  /**
   * @brief Returns the layer type.
   */
  virtual inline const char* type() const {
    return "";
  }
  /**
   * @brief Returns the estimated floating point operations of this layer
   */
  virtual uint_tp ForwardFlops() {
    return 0;
  }

  virtual uint_tp BackwardFlops() {
    return 0;
  }
  /**
   * @brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of bottom blobs.
   */
  virtual inline int_tp ExactNumBottomBlobs() const {
    return -1;
  }
  /**
   * @brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
   */
  virtual inline int_tp MinBottomBlobs() const {
    return -1;
  }
  /**
   * @brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
  virtual inline int_tp MaxBottomBlobs() const {
    return -1;
  }
  /**
   * @brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
  virtual inline int_tp ExactNumTopBlobs() const {
    return -1;
  }
  /**
   * @brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
  virtual inline int_tp MinTopBlobs() const {
    return -1;
  }
  /**
   * @brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
  virtual inline int_tp MaxTopBlobs() const {
    return -1;
  }
  /**
   * @brief Returns true if the layer requires an equal number of bottom and
   *        top blobs.
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
  virtual inline bool EqualNumBottomTopBlobs() const {
    return false;
  }

  /**
   * @brief Return whether "anonymous" top blobs are created automatically
   *        by the layer.
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
  virtual inline bool AutoTopBlobs() const {
    return false;
  }

  /**
   * @brief Return whether to allow force_backward for a given bottom blob
   *        index.
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to blob i only if it needs gradient information
   * (as is done when force_backward == false).
   */
  virtual inline bool AllowForceBackward(const int_tp bottom_index) const {
    return true;
  }

  /**
   * @brief Specifies whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *
   * You can safely ignore false values and always compute gradients
   * for all parameters, but possibly with wasteful computation.
   */
  inline bool param_propagate_down(const int_tp param_id) {
    return
        (param_propagate_down_.size() > param_id) ?
            param_propagate_down_[param_id] : false;
  }
  /**
   * @brief Sets whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   */
  inline void set_param_propagate_down(const int_tp param_id,
                                       const bool value) {
    if (param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }
  /**
   * @brief Writes the layer parameter to a protocol buffer
   */
  virtual void ToProto(LayerParameter* param, bool write_diff = false) = 0;

  virtual int_tp blobs_size() = 0;

  virtual void SetUp(const vector<BlobBase*>& bottom,
        const vector<BlobBase*>& top) = 0;

  /**
   * @brief Returns the vector of learnable parameter blobs.
   */
  virtual vector<shared_ptr<BlobBase> > blob_bases() = 0;

  /**
   * @brief Returns the vector of all initialized quantizers in the layers.
   */
  virtual vector<shared_ptr<QuantizerBase> > get_all_quantizers() = 0;

 protected:
  LayerBase(const LayerParameter& param) {
    layer_param_ = param;
    device_ = Caffe::GetDevice(layer_param_.device(), true);
    // Set phase and copy blobs (if there are any).
    phase_ = param.phase();
  }

  /** Device context */
  Device *device_;
  /** The protobuf that stores the layer parameters */
  LayerParameter layer_param_;
  /** The phase: TRAIN or TEST */
  Phase phase_;
  /** Vector indicating whether to compute the diff of each param blob. */
  vector<bool> param_propagate_down_;

  shared_ptr<QuantizerBase> net_quant_;
  shared_ptr<QuantizerBase> blobs_quant_;
};

/**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Layers must implement a Forward function, in which they take their input
 * (bottom) Blobs (if any) and compute their output Blobs (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blobs, given the error gradients with
 * their output Blobs.
 */
template<typename Dtype, typename MItype, typename MOtype>
class Layer : public LayerBase {
 public:
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   */
  explicit Layer(const LayerParameter& param) : LayerBase(param) {
    if (layer_param_.has_net_quantizer()) {
      net_quant_ = CreateQuantizer(layer_param_.net_quantizer());
    } else {
      QuantizerParameter quant_param;
      quant_param.set_device(this->device_->id());
      quant_param.set_input_data_type(proto_data_type<float>());
      quant_param.set_output_data_type(layer_param_.top_data_type());
      quant_param.set_index(layer_param_.quantizer_index());
      net_quant_ = make_shared<Quantizer<float, MOtype> >(quant_param);
    }
    if (layer_param_.has_blobs_quantizer()) {
      blobs_quant_ = CreateQuantizer(layer_param_.blobs_quantizer());
    } else {
      QuantizerParameter quant_param;
      quant_param.set_device(this->device_->id());
      quant_param.set_input_data_type(proto_data_type<float>());
      quant_param.set_output_data_type(layer_param_.compute_data_type());
      quant_param.set_index(layer_param_.quantizer_index());
      blobs_quant_ = make_shared<Quantizer<float, Dtype> >(quant_param);
    }
    if (layer_param_.has_bottom_quantizer()) {
      bottom_quant_ = make_shared<Quantizer<MItype, Dtype> >(
          layer_param_.bottom_quantizer());
    } else {
      QuantizerParameter quant_param;
      quant_param.set_device(this->device_->id());
      quant_param.set_input_data_type(layer_param_.bottom_data_type());
      quant_param.set_output_data_type(layer_param_.compute_data_type());
      quant_param.set_index(layer_param_.quantizer_index());
      bottom_quant_ = make_shared<Quantizer<MItype, Dtype> >(quant_param);
    }
    if (layer_param_.has_top_quantizer()) {
      top_quant_ = make_shared<Quantizer<Dtype, MOtype> >(
          layer_param_.top_quantizer());
    } else {
      QuantizerParameter quant_param;
      quant_param.set_device(this->device_->id());
      quant_param.set_input_data_type(layer_param_.compute_data_type());
      quant_param.set_output_data_type(layer_param_.top_data_type());
      quant_param.set_index(layer_param_.quantizer_index());
      top_quant_ = make_shared<Quantizer<Dtype, MOtype> >(quant_param);
    }

    if (layer_param_.blobs_size() > 0) {
      blobs_.resize(layer_param_.blobs_size());
      for (int_tp i = 0; i < layer_param_.blobs_size(); ++i) {
        blobs_[i].reset(new Blob<Dtype>(device_));
        blobs_[i]->FromProto(layer_param_.blobs(i));
      }
    }
  }

  virtual ~Layer() {

  }

  /**
   * @brief Implements common layer setup functionality.
   *
   * @param bottom the preshaped input blobs
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape
   *
   * Checks that the number of bottom and top blobs is correct.
   * Calls LayerSetUp to do special layer setup for individual layer types,
   * followed by Reshape to set up sizes of top blobs and internal buffers.
   * Sets up the loss weight multiplier blobs for any non-zero loss weights.
   * This method may not be overridden.
   */
  virtual void SetUp(const vector<BlobBase*>& bottom,
      const vector<BlobBase*>& top) {
    vector<Blob<MItype>*> cast_bottom;
    vector<Blob<MOtype>*> cast_top;
    for (size_t i = 0; i < bottom.size(); ++i) {
      cast_bottom.push_back(static_cast<Blob<MItype>*>(bottom[i]));
    }
    for (size_t i = 0; i < top.size(); ++i) {
      cast_top.push_back(static_cast<Blob<MOtype>*>(top[i]));
    }
    SetUp(cast_bottom, cast_top);
  }

  void SetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
    CheckBlobCounts(bottom, top);
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    SetLossWeights(top);
  }

  /**
   * @brief Does layer-specific setup: your layer should implement this function
   *        as well as Reshape.
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for
   *     this layer
   * @param top
   *     the allocated but unshaped output blobs
   *
   * This method should do one-time layer specific setup. This includes reading
   * and processing relevent parameters from the <code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top blob sizes.
   */
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
                          const vector<Blob<MOtype>*>& top) {
  }

  virtual void Reshape(const vector<BlobBase*>& bottom,
                       const vector<BlobBase*>& top) {
    vector<Blob<MItype>*> cast_bottom;
    vector<Blob<MOtype>*> cast_top;
    for (size_t i = 0; i < bottom.size(); ++i) {
      cast_bottom.push_back(static_cast<Blob<MItype>*>(bottom[i]));
    }
    for (size_t i = 0; i < bottom.size(); ++i) {
      cast_top.push_back(static_cast<Blob<MOtype>*>(top[i]));
    }
    this->Reshape(cast_bottom, cast_top);
  }

  /**
   * @brief Adjust the shapes of top blobs and internal buffers to accommodate
   *        the shapes of the bottom blobs.
   *
   * @param bottom the input blobs, with the requested input shapes
   * @param top the top blobs, which should be reshaped as needed
   *
   * This method should reshape top blobs as needed according to the shapes
   * of the bottom (input) blobs, as well as reshaping any internal buffers
   * and making any other necessary adjustments so that the layer can
   * accommodate the bottom blobs.
   */
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
                       const vector<Blob<MOtype>*>& top) = 0;

  virtual void Forward(const vector<BlobBase*>& bottom,
                       const vector<BlobBase*>& top,
                       void* out_loss) {
    vector<Blob<MItype>*> cast_bottom;
    vector<Blob<MOtype>*> cast_top;
    for (size_t i = 0; i < bottom.size(); ++i) {
      cast_bottom.push_back(static_cast<Blob<MItype>*>(bottom[i]));
    }
    for (size_t i = 0; i < top.size(); ++i) {
      cast_top.push_back(static_cast<Blob<MOtype>*>(top[i]));
    }
    Dtype layer_loss = this->Forward(cast_bottom, cast_top);
    if (out_loss != nullptr) {
      net_quant_->Backward_cpu(1, static_cast<void*>(&layer_loss), out_loss);
    }
  }

  virtual void Backward(const vector<BlobBase*>& top,
                       const vector<bool>& propagate_down,
                       const vector<BlobBase*>& bottom) {
    vector<Blob<MItype>*> cast_bottom;
    vector<Blob<MOtype>*> cast_top;
    for (size_t i = 0; i < bottom.size(); ++i) {
      cast_bottom.push_back(static_cast<Blob<MItype>*>(bottom[i]));
    }
    for (size_t i = 0; i < top.size(); ++i) {
      cast_top.push_back(static_cast<Blob<MOtype>*>(top[i]));
    }
    this->Backward(cast_top, propagate_down, cast_bottom);
  }

  /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     outputs
   * \return The total loss from the layer.
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
   */
  inline Dtype Forward(const vector<Blob<MItype>*>& bottom,
                       const vector<Blob<MOtype>*>& top);

  /**
   * @brief Given the top blob error gradients, compute the bottom blob error
   *        gradients.
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error
   *     with respect to themselves
   * @param propagate_down
   *     a vector with equal length to bottom, with each index indicating
   *     whether to propagate the error gradients down to the bottom blob at
   *     the corresponding index
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error
   *     with respect to themselves after Backward is run
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   *
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
   */
  inline void Backward(const vector<Blob<MOtype>*>& top,
                       const vector<bool>& propagate_down,
                       const vector<Blob<MItype>*>& bottom);

  /**
   * @brief Returns the vector of learnable parameter blobs.
   */
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

  /**
   * @brief Returns the vector of learnable parameter blobs.
   */
  virtual vector<shared_ptr<BlobBase> > blob_bases() {
    vector<shared_ptr<BlobBase> > blob_base_vec;
    for (size_t i = 0; i < blobs_.size(); ++i) {
      blob_base_vec.push_back(blobs_[i]);
    }
    return blob_base_vec;
  }

  virtual vector<shared_ptr<QuantizerBase> > get_all_quantizers() {
    vector<shared_ptr<QuantizerBase> > quant_base_vec;
    if (this->net_quant_ != nullptr) {
      quant_base_vec.push_back(this->net_quant_);
    }
    if (this->blobs_quant_ != nullptr) {
      quant_base_vec.push_back(this->blobs_quant_);
    }
    if (this->bottom_quant_ != nullptr) {
      quant_base_vec.push_back(this->bottom_quant_);
    }
    if (this->top_quant_ != nullptr) {
      quant_base_vec.push_back(this->top_quant_);
    }
    return quant_base_vec;
  }


  /**
   * @brief Returns the scalar loss associated with a top blob at a given index.
   */
  inline Dtype loss(const int_tp top_index) const {
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
  }

  virtual void loss(const int_tp top_index, void* out_loss) {
    Dtype layer_loss = loss(top_index);
    net_quant_->Backward_cpu(1, static_cast<void*>(&layer_loss), out_loss);
  }

  /**
   * @brief Sets the loss associated with a top blob at a given index.
   */
  inline void set_loss(const int_tp top_index, const Dtype value) {
    if (loss_.size() <= top_index) {
      loss_.resize(top_index + 1, Dtype(0));
    }
    loss_[top_index] = value;
  }

  virtual int_tp blobs_size() {
    return blobs_.size();
  }

 protected:
  /** The vector that stores the learnable parameters as a set of blobs. */
  vector<shared_ptr<Blob<Dtype> > > blobs_;

  /** The vector that indicates whether each top blob has a non-zero weight in
   *  the objective function. */
  vector<Dtype> loss_;

  /** Quantizers */
  shared_ptr<Quantizer<MItype, Dtype> > bottom_quant_;
  shared_ptr<Quantizer<Dtype, MOtype> > top_quant_;

  /** Device program */
  shared_ptr<DeviceProgram> device_program_;

  /** @brief Using the CPU device, compute the layer output. */
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
                           const vector<Blob<MOtype>*>& top) = 0;
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   */
  virtual void Forward_gpu(const vector<Blob<MItype>*>& bottom,
                           const vector<Blob<MOtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Forward_cpu(bottom, top);
  }

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   */
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<MItype>*>& bottom) = 0;
  /**
   * @brief Using the GPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
   */
  virtual void Backward_gpu(const vector<Blob<MOtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<MItype>*>& bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom);
  }

  /**
   * @brief Writes the layer parameter to a protocol buffer
   */
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

  /**
   * Called by the parent Layer's SetUp to check that the number of bottom
   * and top Blobs provided as input match the expected numbers specified by
   * the {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
   */
  virtual void CheckBlobCounts(const vector<Blob<MItype>*>& bottom,
                               const vector<Blob<MOtype>*>& top) {
    if (ExactNumBottomBlobs() >= 0) {
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())<< type()
          << " Layer takes " << ExactNumBottomBlobs()
      << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
      CHECK_LE(MinBottomBlobs(), bottom.size())
      << type() << " Layer takes at least " << MinBottomBlobs()
      << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
      CHECK_GE(MaxBottomBlobs(), bottom.size())
      << type() << " Layer takes at most " << MaxBottomBlobs()
      << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
      CHECK_EQ(ExactNumTopBlobs(), top.size())
      << type() << " Layer produces " << ExactNumTopBlobs()
      << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
      CHECK_LE(MinTopBlobs(), top.size())
      << type() << " Layer produces at least " << MinTopBlobs()
      << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
      CHECK_GE(MaxTopBlobs(), top.size())
      << type() << " Layer produces at most " << MaxTopBlobs()
      << " top blob(s) as output.";
    }
    if (EqualNumBottomTopBlobs()) {
      CHECK_EQ(bottom.size(), top.size())
      << type() << " Layer produces one top blob as output for each "
      << "bottom blob input.";
    }
  }

  /**
   * Called by SetUp to initialize the weights associated with any top blobs in
   * the loss function. Store non-zero loss weights in the diff blob.
   */
  inline void SetLossWeights(const vector<Blob<MOtype>*>& top) {
    const int_tp num_loss_weights = layer_param_.loss_weight_size();
    if (num_loss_weights) {
      CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
      "unspecified or specified once per top blob.";
      for (int_tp top_id = 0; top_id < top.size(); ++top_id) {
        float loss_weight_param = layer_param_.loss_weight(top_id);
        MOtype loss_weight = MOtype(0);
        Quantizer<float, MOtype> quant(net_quant_->quant_param());
        quant.Forward_cpu(1, &loss_weight_param, &loss_weight);
        if (loss_weight == MOtype(0)) { continue; }
        this->set_loss(top_id, loss_weight);
        const int_tp count = top[top_id]->count();
        MOtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
        caffe_set(count, loss_weight, loss_multiplier);
      }
    }
  }

 private:
  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template<typename Dtype, typename MItype, typename MOtype>
inline Dtype Layer<Dtype, MItype, MOtype>::Forward(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  Dtype loss = 0;
  Reshape(bottom, top);
  switch (Caffe::mode()) {
    case Caffe::CPU:
      for (int_tp bottom_id = 0; bottom_id < bottom.size(); ++bottom_id) {
        bottom_quant_->Observe_in_cpu(bottom[bottom_id]->count(),
                                      bottom[bottom_id]->cpu_data());
      }
      Forward_cpu(bottom, top);
      for (int_tp top_id = 0; top_id < top.size(); ++top_id) {
        top_quant_->Observe_out_cpu(top[top_id]->count(),
                                    top[top_id]->cpu_data());
        if (!this->loss(top_id)) {
          continue;
        }
        const int_tp count = top[top_id]->count();
        const MOtype* data = top[top_id]->cpu_data();
        const MOtype* loss_weights = top[top_id]->cpu_diff();
        loss += caffe_dot(count, data, loss_weights);
      }
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      for (int_tp bottom_id = 0; bottom_id < bottom.size(); ++bottom_id) {
        bottom_quant_->Observe_in_gpu(bottom[bottom_id]->count(),
                                  bottom[bottom_id]->gpu_data());
      }
      #endif  // !CPU_ONLY
      Forward_gpu(bottom, top);
#ifndef CPU_ONLY
      for (int_tp top_id = 0; top_id < top.size(); ++top_id) {
        top_quant_->Observe_out_gpu(top[top_id]->count(),
                                    top[top_id]->gpu_data());
        if (!this->loss(top_id)) {
          continue;
        }
        const int_tp count = top[top_id]->count();
        vptr<const MOtype> data = top[top_id]->gpu_data();
        vptr<const MOtype> loss_weights = top[top_id]->gpu_diff();
        MOtype blob_loss = 0;
        device_->template dot<MOtype>(count, data, loss_weights, &blob_loss);
        // TODO: Type conversion may be necessary
        loss += blob_loss;
      }
#endif  // !CPU_ONLY
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
  return loss;
}

template<typename Dtype, typename MItype, typename MOtype>
inline void Layer<Dtype, MItype, MOtype>::Backward(
                      const vector<Blob<MOtype>*>& top,
                      const vector<bool>& propagate_down,
                      const vector<Blob<MItype>*>& bottom) {
  switch (Caffe::mode()) {
    case Caffe::CPU:
      for (int_tp top_id = 0; top_id < top.size(); ++top_id) {
        top_quant_->Observe_out_cpu(top[top_id]->count(),
                                    top[top_id]->cpu_diff());
      }
      Backward_cpu(top, propagate_down, bottom);
      for (int_tp bottom_id = 0; bottom_id < bottom.size(); ++bottom_id) {
        bottom_quant_->Observe_in_cpu(bottom[bottom_id]->count(),
                                  bottom[bottom_id]->cpu_diff());
      }
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      for (int_tp top_id = 0; top_id < top.size(); ++top_id) {
        top_quant_->Observe_out_gpu(top[top_id]->count(),
                                    top[top_id]->gpu_diff());
      }
#endif  // !CPU_ONLY
      Backward_gpu(top, propagate_down, bottom);
#ifndef CPU_ONLY
      for (int_tp bottom_id = 0; bottom_id < bottom.size(); ++bottom_id) {
        bottom_quant_->Observe_in_gpu(bottom[bottom_id]->count(),
                                  bottom[bottom_id]->gpu_diff());
      }
#endif  // !CPU_ONLY
      break;
    default:
      LOG(FATAL)<< "Unknown caffe mode.";
    }
  }

// Serialize LayerParameter to protocol buffer
template<typename Dtype, typename MItype, typename MOtype>
void Layer<Dtype, MItype, MOtype>::ToProto(
    LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int_tp i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
}

}  // namespace caffe

#endif  // CAFFE_LAYER_H_
