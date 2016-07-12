#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "device.hpp"


namespace caffe {

/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
template<typename Dtype>
class Net {
 public:
  explicit Net(const NetParameter& param, device* device_context,
               const Net* root_net =
               NULL);
  explicit Net(const string& param_file, Phase phase, device* device_context,
               const int level = 0, const vector<string>* stages = NULL,
               const Net* root_net = NULL);
  virtual ~Net() {
  }

  /// @brief Initialize a network with a NetParameter.
  void Init(const NetParameter& param);

  /**
   * @brief Run Forward and return the result.
   *
   */
  const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);
  /// @brief DEPRECATED; use Forward() instead.
  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL) {
    LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: ForwardPrefilled() "
        << "will be removed in a future version. Use Forward().";
    return Forward(loss);
  }

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   */
  Dtype ForwardFromTo(int_tp start, int_tp end);
  Dtype ForwardFrom(int_tp start);
  Dtype ForwardTo(int_tp end);
  /// @brief DEPRECATED; set input blobs then use Forward() instead.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL);


  /**
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   */
  void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  void Backward();
  void BackwardFromTo(int_tp start, int_tp end);
  void BackwardFrom(int_tp start);
  void BackwardTo(int_tp end);

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();

  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);
    Backward();
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
  void Update();
  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
  void ShareWeights();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
  void ShareTrainedLayersWith(const Net* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string trained_filename);
  void CopyTrainedLayersFromHDF5(const string trained_filename);
  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false) const;
  /// @brief Writes the net to an HDF5 file.
  void ToHDF5(const string& filename, bool write_diff = false) const;

  /// @brief returns the network name.
  inline const string& name() const {
    return name_;
  }
  /// @brief returns the layer names
  inline const vector<string>& layer_names() const {
    return layer_names_;
  }
  /// @brief returns the blob names
  inline const vector<string>& blob_names() const {
    return blob_names_;
  }
  /// @brief returns the blobs
  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const {
    return blobs_;
  }
  /// @brief returns the layers
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
    return layers_;
  }
  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const {
    return phase_;
  }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
    return top_vecs_;
  }
  /// @brief returns the ids of the top blobs of layer i
  inline const vector<int_tp> & top_ids(int_tp i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
    return top_id_vecs_[i];
  }
  /// @brief returns the ids of the bottom blobs of layer i
  inline const vector<int_tp> & bottom_ids(int_tp i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }
  inline const vector<vector<bool> >& bottom_need_backward() const {
    return bottom_need_backward_;
  }
  inline const vector<Dtype>& blob_loss_weights() const {
    return blob_loss_weights_;
  }
  inline const vector<bool>& layer_need_backward() const {
    return layer_need_backward_;
  }
  /// @brief returns the parameters
  inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
    return params_;
  }
  inline const vector<Blob<Dtype>*>& learnable_params() const {
    return learnable_params_;
  }
  /// @brief returns the learnable parameter learning rate multipliers
  inline const vector<float>& params_lr() const { return params_lr_; }
  inline const vector<bool>& has_params_lr() const { return has_params_lr_; }
  /// @brief returns the learnable parameter decay multipliers
  inline const vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  inline const vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
  const map<string, int_tp>& param_names_index() const {
    return param_names_index_;
  }
  inline const vector<int_tp>& param_owners() const {
    return param_owners_;
  }
  inline const vector<string>& param_display_names() const {
    return param_display_names_;
  }
  /// @brief Input and output blob numbers
  inline int_tp num_inputs() const {
    return net_input_blobs_.size();
  }
  inline int_tp num_outputs() const {
    return net_output_blobs_.size();
  }
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  inline const vector<int_tp>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  inline const vector<int_tp>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  bool has_blob(const string& blob_name) const;
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
  bool has_layer(const string& layer_name) const;
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;

  void set_debug_info(const bool value) {
    debug_info_ = value;
  }

  // Helpers for Init.
  /**
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   */
  static void FilterNet(const NetParameter& param,
                        NetParameter* param_filtered);
  /// @brief return whether NetState state meets NetStateRule rule
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
                             const string& layer_name);

 protected:
  // Helpers for Init.
  /// @brief Append a new top blob to the net.
  void AppendTop(const NetParameter& param, const int_tp layer_id,
                 const int_tp top_id, set<string>* available_blobs,
                 map<string, int_tp>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
  int_tp AppendBottom(const NetParameter& param, const int_tp layer_id,
                   const int_tp bottom_id, set<string>* available_blobs,
                   map<string, int_tp>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
  void AppendParam(const NetParameter& param, const int_tp layer_id,
                   const int_tp param_id);

  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int_tp layer_id);
  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int_tp layer_id);
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int_tp param_id);

  /// @brief The network name
  string name_;
  /// @brief The phase: TRAIN or TEST
  Phase phase_;
  /// @brief Individual layers in the net
  vector<shared_ptr<Layer<Dtype> > > layers_;
  vector<string> layer_names_;
  map<string, int_tp> layer_names_index_;
  vector<bool> layer_need_backward_;
  /// @brief the blobs storing intermediate results between the layer.
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  vector<string> blob_names_;
  map<string, int_tp> blob_names_index_;
  vector<bool> blob_need_backward_;
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  vector<vector<Blob<Dtype>*> > bottom_vecs_;
  vector<vector<int_tp> > bottom_id_vecs_;
  vector<vector<bool> > bottom_need_backward_;
  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<Dtype>*> > top_vecs_;
  vector<vector<int_tp> > top_id_vecs_;
  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.
  vector<Dtype> blob_loss_weights_;
  vector<vector<int_tp> > param_id_vecs_;
  vector<int_tp> param_owners_;
  vector<string> param_display_names_;
  vector<pair<int_tp, int_tp> > param_layer_indices_;
  map<string, int_tp> param_names_index_;
  /// blob indices for the input and the output of the net
  vector<int_tp> net_input_blob_indices_;
  vector<int_tp> net_output_blob_indices_;
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<Blob<Dtype>*> net_output_blobs_;
  /// The parameters in the network.
  vector<shared_ptr<Blob<Dtype> > > params_;
  vector<Blob<Dtype>*> learnable_params_;
  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  vector<int_tp> learnable_param_ids_;
  /// the learning rate multipliers for learnable_params_
  vector<float> params_lr_;
  vector<bool> has_params_lr_;
  /// the weight decay multipliers for learnable_params_
  vector<float> params_weight_decay_;
  vector<bool> has_params_decay_;
  /// The bytes of memory used by this net
  uint_tp memory_used_;
  /// Whether to compute and display debug info for the net.
  bool debug_info_;

  device* device_;

  /// The root net that actually holds the shared layers in data parallelism
  const Net* const root_net_;
  DISABLE_COPY_AND_ASSIGN(Net);
};

}  // namespace caffe

#endif  // CAFFE_NET_HPP_
