#ifdef MKL2017_SUPPORTED
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/mkl_layers.hpp"

namespace caffe {

template <typename Dtype>
MKLBatchNormLayer<Dtype>::~MKLBatchNormLayer()
{
  if (batchNormFwd != NULL) dnnDelete<Dtype>(batchNormFwd);
  if (batchNormBwdData != NULL) dnnDelete<Dtype>(batchNormBwdData);
  if (batchNormBwdScaleShift != NULL) dnnDelete<Dtype>(batchNormBwdScaleShift);

  dnnLayoutDelete<Dtype>(layout_usr_);
  dnnReleaseBuffer<Dtype>(workspace_buffer_);
  dnnReleaseBuffer<Dtype>(scaleShift_buffer_);
}

template <typename Dtype>
void MKLBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  eps_ = this->layer_param_.batch_norm_param().eps();
  use_weight_bias_ = this->layer_param_.batch_norm_param().use_weight_bias();
  bias_term_ = this->layer_param_.batch_norm_param().bias_term();

  size_t dim = 4, sizes[4], strides[4];

  channels_ = bottom[0]->channels();
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();
  num_      = bottom[0]->num();

  sizes[0] = width_;
  sizes[1] = height_;
  sizes[2] = channels_;
  sizes[3] = num_;

  strides[0] = 1;
  strides[1] = sizes[0];
  strides[2] = sizes[0]*sizes[1];
  strides[3] = sizes[0]*sizes[1]*sizes[2];

  dnnError_t e;
  e = dnnLayoutCreate<Dtype>(&layout_usr_, dim, sizes, strides);
  CHECK_EQ(e, E_SUCCESS);

  workspace_buffer_ = NULL;
  scaleShift_buffer_ = NULL;
  // "Lazy" allocation because here we don't know
  // what layout is used by neighbours.
  batchNormFwd = NULL; // Will be allocated in a "lazy" way in first forward pass
  batchNormBwdData = NULL; // Will be allocated in a "lazy" way in first backward pass
  batchNormBwdScaleShift = NULL;  // Will be allocated in a "lazy" way in first backward pass if it is required

  if (use_weight_bias_)
  {
    if ( bias_term_ ) {
        this->blobs_.resize(2);
    } else {
        this->blobs_.resize(1);
    }
    // Initialize scale and shift
    vector<int> scaleshift_shape(1);
    scaleshift_shape[0] = channels_;

    this->blobs_[0].reset(new Blob<Dtype>(scaleshift_shape));
    Dtype* data = this->blobs_[0]->mutable_cpu_data();
    FillerParameter filler_param(this->layer_param_.batch_norm_param().filler());
    if (!this->layer_param_.batch_norm_param().has_filler()) {
      filler_param.set_type("constant");
      filler_param.set_value(1);
    }
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
    filler->Fill(this->blobs_[0].get());

    if ( bias_term_ ) {
      this->blobs_[1].reset(new Blob<Dtype>(scaleshift_shape));
      Dtype* data = this->blobs_[1]->mutable_cpu_data();
      FillerParameter bias_filler_param(this->layer_param_.batch_norm_param().bias_filler());
      if (!this->layer_param_.batch_norm_param().has_bias_filler()) {
        bias_filler_param.set_type("constant");
        bias_filler_param.set_value(0);
      }
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(bias_filler_param));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
}


template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_ = bottom[0]->num();
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  void* bottom_data = (void*)bottom[0]->prv_data();
  void* top_data = NULL;
  int is_first_pass = 0;

  if(NULL != bottom_data)
  {
    // Is it the first pass? Create a primitive.
    if (batchNormFwd == NULL) {
      is_first_pass = 1;

      CHECK((bottom[0]->get_prv_descriptor_data())->get_descr_type() == PrvMemDescr::PRV_DESCR_MKL2017);
      shared_ptr<MKLData<Dtype> > mem_descr
        =  boost::static_pointer_cast<MKLData<Dtype> > (bottom[0]->get_prv_descriptor_data());
      CHECK(mem_descr != NULL);

      dnnError_t e;

      e = dnnBatchNormalizationCreateForward<Dtype>(&batchNormFwd, NULL, mem_descr->layout_int, eps_);
      CHECK_EQ(e, E_SUCCESS);

      fwd_top_data = mem_descr;
    }
    top_data = top[0]->mutable_prv_data();
    top[0]->set_prv_descriptor_data(fwd_top_data);

  } else {
    DLOG(INFO) << "Using cpu_data in MKLBatchNormLayer.";
    if (batchNormFwd == NULL) {
      // First pass
      is_first_pass = 1;

      dnnError_t e;
      e = dnnBatchNormalizationCreateForward<Dtype>(&batchNormFwd, NULL, layout_usr_, eps_);
      CHECK_EQ(e, E_SUCCESS);
    }
    bottom_data = (void*)bottom[0]->cpu_data();
    top_data = top[0]->mutable_cpu_data();
  }
  if (is_first_pass == 1) {
      dnnError_t e;

      dnnLayout_t workspace_buffer_l = NULL;
      e = dnnLayoutCreateFromPrimitive<Dtype>(&workspace_buffer_l, batchNormFwd, dnnResourceWorkspace);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnAllocateBuffer<Dtype>((void **)&workspace_buffer_, workspace_buffer_l);
      CHECK_EQ(e, E_SUCCESS);
      dnnLayoutDelete<Dtype>(workspace_buffer_l);

      dnnLayout_t scaleShift_buffer_l = NULL;
      e = dnnLayoutCreateFromPrimitive<Dtype>(&scaleShift_buffer_l, batchNormFwd, dnnResourceScaleShift);
      CHECK_EQ(e, E_SUCCESS);
      e = dnnAllocateBuffer<Dtype>((void **)&scaleShift_buffer_, scaleShift_buffer_l);
      CHECK_EQ(e, E_SUCCESS);
      dnnLayoutDelete<Dtype>(scaleShift_buffer_l);
      if (!use_weight_bias_)
      {
         for (int i = 0; i < channels_; i++) {
            scaleShift_buffer_[i] = 1.0;
            scaleShift_buffer_[channels_ + i] = 0;
         }
      }
  }

  if (use_weight_bias_) {
    // Fill ScaleShift buffer
    for (int i = 0; i < channels_; i++) {
      scaleShift_buffer_[i] = this->blobs_[0]->cpu_data()[i];
      scaleShift_buffer_[channels_ + i] = 0;
      if (bias_term_) {
         scaleShift_buffer_[channels_ + i] = this->blobs_[1]->cpu_data()[i];
      }
    }
  }
  dnnError_t e;
  void* BatchNorm_res[dnnResourceNumber];
  BatchNorm_res[dnnResourceSrc] = bottom_data;
  BatchNorm_res[dnnResourceDst] = top_data;
  BatchNorm_res[dnnResourceWorkspace] = workspace_buffer_;
  BatchNorm_res[dnnResourceScaleShift] = scaleShift_buffer_;

  e = dnnExecute<Dtype>(batchNormFwd, BatchNorm_res);
  CHECK_EQ(e, E_SUCCESS);
}

template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  void* top_diff = (void*)top[0]->prv_diff();
  void* bottom_data = (void*)bottom[0]->prv_data();
  void* bottom_diff = NULL;

  if (top_diff && bottom_data) {
    bottom_diff = (void*)bottom[0]->mutable_prv_diff();
    // Is it the first pass? Create a primitive.
    if (batchNormBwdData == NULL) {
      CHECK((top[0]->get_prv_descriptor_diff())->get_descr_type() == PrvMemDescr::PRV_DESCR_MKL2017);
      shared_ptr<MKLDiff<Dtype> > mem_descr
        =  boost::static_pointer_cast<MKLDiff<Dtype> > (top[0]->get_prv_descriptor_diff());
      CHECK(mem_descr != NULL);

      dnnError_t e;
      e = dnnBatchNormalizationCreateBackwardData<Dtype>(&batchNormBwdData, NULL, mem_descr->layout_int, eps_);
      CHECK_EQ(e, E_SUCCESS);

      bwd_bottom_diff = mem_descr;

      if (use_weight_bias_) {
        e = dnnBatchNormalizationCreateBackwardScaleShift<Dtype>(&batchNormBwdScaleShift, NULL, mem_descr->layout_int, eps_);
      }
    }
    bottom[0]->set_prv_descriptor_diff(bwd_bottom_diff);

  } else {
    DLOG(INFO) << "Using cpu_data in MKLBatchNormLayer.";
    top_diff = (void*)top[0]->cpu_diff();
    bottom_data = (void*)bottom[0]->cpu_data();
    bottom_diff = (void*)bottom[0]->mutable_cpu_diff();
    if (batchNormBwdData == NULL) {
      dnnError_t e;
      e = dnnBatchNormalizationCreateBackwardData<Dtype>(&batchNormBwdData, NULL, layout_usr_, eps_);
      CHECK_EQ(e, E_SUCCESS);
      if (use_weight_bias_) {
        e = dnnBatchNormalizationCreateBackwardScaleShift<Dtype>(&batchNormBwdScaleShift, NULL, layout_usr_, eps_);
        CHECK_EQ(e, E_SUCCESS);
      }
    }
  }

  dnnError_t e;
  void* BatchNorm_res[dnnResourceNumber];
  BatchNorm_res[dnnResourceSrc] = bottom_data;
  BatchNorm_res[dnnResourceDiffDst] = top_diff;
  BatchNorm_res[dnnResourceDiffSrc] = bottom_diff;
  BatchNorm_res[dnnResourceWorkspace] = workspace_buffer_;
  BatchNorm_res[dnnResourceScaleShift] = scaleShift_buffer_;

  e = dnnExecute<Dtype>(batchNormBwdData, BatchNorm_res);
  CHECK_EQ(e, E_SUCCESS);

  if (use_weight_bias_) {
    void* BatchNormBwdScaleShift_res[dnnResourceNumber];
    BatchNormBwdScaleShift_res[dnnResourceSrc] = bottom_data;
    BatchNormBwdScaleShift_res[dnnResourceDiffDst] = top_diff;
    BatchNormBwdScaleShift_res[dnnResourceDiffSrc] = bottom_diff;
    BatchNormBwdScaleShift_res[dnnResourceWorkspace] = workspace_buffer_;
    BatchNormBwdScaleShift_res[dnnResourceDiffScaleShift] = scaleShift_buffer_;
    e = dnnExecute<Dtype>(batchNormBwdScaleShift, BatchNormBwdScaleShift_res);
    CHECK_EQ(e, E_SUCCESS);
    // Store ScaleShift blobs
    Dtype* diff_scale = this->blobs_[0]->mutable_cpu_diff();
    Dtype* diff_shift = this->blobs_[1]->mutable_cpu_diff();
    for (int i = 0; i < channels_; i++) {
      diff_scale[i] =  scaleShift_buffer_[i];
      diff_shift[i] =  0;
      if (bias_term_) {
         diff_shift[i] =  scaleShift_buffer_[channels_ + i];
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(MKLBatchNormLayer);
#else
template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLBatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKLBatchNormLayer);
//REGISTER_LAYER_CLASS(MKLBatchNorm);
}  // namespace caffe
#endif //#ifdef MKL2017_SUPPORTED
