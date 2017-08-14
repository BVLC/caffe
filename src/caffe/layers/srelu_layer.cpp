#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
// param: 0: thresh; 1: pslope
template <typename Dtype>
void SReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  SReLUParameter srelu_param = this->layer_param().srelu_param();
  int channels = bottom[0]->channels();
  negative_slope_ = srelu_param.negative_slope();
  thresh_channel_shared_  = srelu_param.thresh_channel_shared();
  pslope_channel_shared_  = srelu_param.pslope_channel_shared();
  nslope_channel_shared_  = srelu_param.nslope_channel_shared();
  nthresh_channel_shared_ = srelu_param.nthresh_channel_shared();
  // if (thresh_channel_shared_ || pslope_channel_shared_ ||
  //     nslope_channel_shared_ || nthresh_channel_shared_)
  //   LOG(FATAL) << "Error of share";
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(4);
    if (thresh_channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, channels));
    }
    if (pslope_channel_shared_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, 1));
    } else {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, channels));
    }
    if (nslope_channel_shared_) {
      this->blobs_[2].reset(new Blob<Dtype>(1, 1, 1, 1));
    } else {
      this->blobs_[2].reset(new Blob<Dtype>(1, 1, 1, channels));
    }
    if (nthresh_channel_shared_) {
      this->blobs_[3].reset(new Blob<Dtype>(1, 1, 1, 1));
    } else {
      this->blobs_[3].reset(new Blob<Dtype>(1, 1, 1, channels));
    }
    shared_ptr<Filler<Dtype> >  thresh_filler;
    shared_ptr<Filler<Dtype> >  pslope_filler;
    shared_ptr<Filler<Dtype> >  nslope_filler;
    shared_ptr<Filler<Dtype> > nthresh_filler;
    // filler for thresh
    if (srelu_param.has_thresh_filler()) {
      thresh_filler.reset(GetFiller<Dtype>(srelu_param.thresh_filler()));
    } else {
      FillerParameter thresh_filler_param;
      thresh_filler_param.set_type("constant");
      thresh_filler_param.set_value(40.);
      thresh_filler.reset(GetFiller<Dtype>(thresh_filler_param));
    }
    thresh_filler->Fill(this->blobs_[0].get());
    // filler for pslope
    if (srelu_param.has_pslope_filler()) {
      pslope_filler.reset(GetFiller<Dtype>(srelu_param.pslope_filler()));
    } else {
      FillerParameter pslope_filler_param;
      pslope_filler_param.set_type("constant");
      pslope_filler_param.set_value(1.);
      pslope_filler.reset(GetFiller<Dtype>(pslope_filler_param));
    }
    pslope_filler->Fill(this->blobs_[1].get());
    // filler for nslope
    if (srelu_param.has_nslope_filler()) {
      nslope_filler.reset(GetFiller<Dtype>(srelu_param.nslope_filler()));
    } else {
      FillerParameter nslope_filler_param;
      nslope_filler_param.set_type("constant");
      nslope_filler_param.set_value(0.20);
      nslope_filler.reset(GetFiller<Dtype>(nslope_filler_param));
    }
    nslope_filler->Fill(this->blobs_[2].get());
    // filler for nthresh
    if (srelu_param.has_nthresh_filler()) {
      nthresh_filler.reset(GetFiller<Dtype>(srelu_param.nthresh_filler()));
    } else {
      FillerParameter nthresh_filler_param;
      nthresh_filler_param.set_type("constant");
      nthresh_filler_param.set_value(0.);
      nthresh_filler.reset(GetFiller<Dtype>(nthresh_filler_param));
    }
    nthresh_filler->Fill(this->blobs_[3].get());
  }

  if (thresh_channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Thresh size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Thresh size is inconsistent with prototxt config";
  }
  if (pslope_channel_shared_) {
    CHECK_EQ(this->blobs_[1]->count(), 1)
        << "Pslope size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[1]->count(), channels)
        << "Pslope size is inconsistent with prototxt config";
  }
  if (nslope_channel_shared_) {
    CHECK_EQ(this->blobs_[2]->count(), 1)
        << "Nslope size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[2]->count(), channels)
        << "Nslope size is inconsistent with prototxt config";
  }
  if (nthresh_channel_shared_) {
    CHECK_EQ(this->blobs_[3]->count(), 1)
      << "Nthresh size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[3]->count(), channels)
      << "Nthresh size is inconsistent with prototxt config";
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_.Reshape(1, 1, 1, bottom[0]->count() / bottom[0]->num());
   thresh_backward_buff_.Reshape(1, 1, 1, bottom[0]->count() / bottom[0]->num());
   pslope_backward_buff_.Reshape(1, 1, 1, bottom[0]->count() / bottom[0]->num());
   nslope_backward_buff_.Reshape(1, 1, 1, bottom[0]->count() / bottom[0]->num());
  nthresh_backward_buff_.Reshape(1, 1, 1, bottom[0]->count() / bottom[0]->num());
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void SReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
//  CHECK_GE(bottom[0]->num_axes(), 2)
//      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
}

// !!! No code for nslope
template <typename Dtype>
void SReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->height() * bottom[0]->width();
  const int channels = bottom[0]->channels();
  const Dtype* thresh_data = this->blobs_[0]->cpu_data();
  const Dtype* pslope_data = this->blobs_[1]->cpu_data();

  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int thresh_div_factor = thresh_channel_shared_ ? channels : 1;
  const int pslope_div_factor = pslope_channel_shared_ ? channels : 1;
  for (int i = 0; i < count; ++i) {
    int th_c = (i / dim) % channels / thresh_div_factor;
    int ps_c = (i / dim) % channels / pslope_div_factor;
    if (bottom_data[i] <= 0)
      top_data[i] = negative_slope_ * bottom_data[i];
    else if (bottom_data[i] < thresh_data[th_c])
      top_data[i] = bottom_data[i];
    else
      top_data[i] = thresh_data[i] +
                    pslope_data[ps_c] * (bottom_data[i] - thresh_data[th_c]);
  }
}

// !!! No code for nslope
template <typename Dtype>
void SReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* thresh_data = this->blobs_[0]->cpu_data();
  const Dtype* pslope_data = this->blobs_[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->height() * bottom[0]->width();
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int thresh_div_factor = thresh_channel_shared_ ? channels : 1;
  const int pslope_div_factor = pslope_channel_shared_ ? channels : 1;
  Dtype* thresh_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* pslope_diff = this->blobs_[1]->mutable_cpu_diff();
  // Propagte to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    for (int i = 0; i < count; ++i) {
      int th_c = (i / dim) % channels / thresh_div_factor;
      int ps_c = (i / dim) % channels / pslope_div_factor;
      if (bottom_data[i] >= thresh_data[th_c]) {
        thresh_diff[th_c] += top_diff[i] * (1. - pslope_data[ps_c]);
        pslope_diff[ps_c] += top_diff[i] * (bottom_data[i] - thresh_data[th_c]);
      } else {
        thresh_diff[th_c] += 0.;
        pslope_diff[ps_c] += 0.;
      }
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int th_c = (i / dim) % channels / thresh_div_factor;
      int ps_c = (i / dim) % channels / pslope_div_factor;
      if (bottom_data[i] <= 0)
        bottom_diff[i] = negative_slope_ * top_diff[i];
      else if (bottom_data[i] < thresh_data[th_c])
        bottom_diff[i] = top_diff[i];
      else
        bottom_diff[i] = pslope_data[ps_c] * top_diff[i];
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SReLULayer);
#endif

INSTANTIATE_CLASS(SReLULayer);
REGISTER_LAYER_CLASS(SReLU);
}  // namespace caffe
