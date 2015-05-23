#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DummyDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_top = top.size();
  const DummyDataParameter& param = this->layer_param_.dummy_data_param();
  const int num_data_filler = param.data_filler_size();
  CHECK(num_data_filler == 0 || num_data_filler == 1 ||
        num_data_filler == num_top)
      << "Number of data fillers must be 0, 1 or equal to the number of tops: "
      << num_top << "; you specified " << num_data_filler << " data fillers.";

  const bool legacy_dims = param.num_size() || param.channels_size() ||
                           param.height_size() || param.width_size();
  if (legacy_dims) {
    CHECK_EQ(0, param.shape_size())
        << "Both shape and legacy fields were specified";
    // Using deprecated 4D output dim specifiers.
    CHECK(param.num_size() == 1 || param.num_size() == num_top)
        << "Must specify 'num' once, or once per top blob "
        << "(" << num_top << "); specified " << param.num_size() << ".";
    CHECK(param.channels_size() == 1 || param.channels_size() == num_top)
        << "Must specify 'channels' once, or once per top blob "
        << "(" << num_top << "); specified " << param.channels_size() << ".";
    CHECK(param.height_size() == 1 || param.height_size() == num_top)
        << "Must specify 'height' once, or once per top blob "
        << "(" << num_top << "); specified " << param.height_size() << ".";
    CHECK(param.width_size() == 1 || param.width_size() == num_top)
        << "Must specify 'width' once, or once per top blob "
        << "(" << num_top << "); specified " << param.width_size() << ".";
  } else {
    CHECK(param.shape_size() == 1 || param.shape_size() == num_top)
        << "Must specify 'shape' once, or once per top blob "
        << "(" << num_top << "); specified " << param.shape_size() << ".";
  }
  // refill_[i] tells Forward i whether or not to actually refill top Blob i.
  // If refill_[i] is false, Forward does nothing for Blob i. We use this to
  // avoid wastefully refilling "constant" Blobs in every forward pass.
  // We first fill refill_ in with the INVERSE of its final values.
  // The first time we run Forward from the LayerSetUp method, we'll fill only
  // Blobs for which refill_ is normally false.  These Blobs will never be
  // filled again.
  refill_.clear();
  fillers_.clear();
  if (num_data_filler <= 1) {
    FillerParameter filler_param;
    if (num_data_filler == 0) {
      filler_param.set_type("constant");
      filler_param.set_value(0);
    } else {
      filler_param.CopyFrom(param.data_filler(0));
    }
    // Refill on each iteration iff not using a constant filler,
    // but use the inverse of this rule for the first run.
    refill_.resize(1);
    refill_[0] = (strcmp(filler_param.type().c_str(), "constant") == 0);
    fillers_.resize(1);
    fillers_[0].reset(GetFiller<Dtype>(filler_param));
  } else {
    refill_.resize(num_top);
    fillers_.resize(num_top);
    for (int i = 0; i < num_top; ++i) {
      fillers_[i].reset(GetFiller<Dtype>(param.data_filler(i)));
      // Refill on each iteration iff not using a constant filler,
      // but use the inverse of this rule for the first run.
      refill_[i] =
          (strcmp(param.data_filler(i).type().c_str(), "constant") == 0);
    }
  }
  for (int i = 0; i < num_top; ++i) {
    if (legacy_dims) {
      const int num = (param.num_size() == 1) ? param.num(0) : param.num(i);
      const int channels =
          (param.channels_size() == 1) ? param.channels(0) : param.channels(i);
      const int height =
          (param.height_size() == 1) ? param.height(0) : param.height(i);
      const int width =
          (param.width_size() == 1) ? param.width(0) : param.width(i);
      top[i]->Reshape(num, channels, height, width);
    } else {
      const int shape_index = (param.shape_size() == 1) ? 0 : i;
      top[i]->Reshape(param.shape(shape_index));
    }
  }
  // Run Forward once, with refill_ inverted, to fill the constant Blobs.
  this->Forward(bottom, top);
  // Invert the inverted refill_ values to refill the desired (non-constant)
  // Blobs in every usual forward pass.
  for (int i = 0; i < refill_.size(); ++i) {
    refill_[i] = !refill_[i];
  }
}

template <typename Dtype>
void DummyDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    const int filler_id = (fillers_.size() > 1) ? i : 0;
    if (refill_[filler_id]) {
      fillers_[filler_id]->Fill(top[i]);
    }
  }
}

INSTANTIATE_CLASS(DummyDataLayer);
REGISTER_LAYER_CLASS(DummyData);

}  // namespace caffe
