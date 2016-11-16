#if defined(MKL2017_SUPPORTED) && defined(USE_MKL2017_NEW_API)
#include <cfloat>
#include <vector>

#include "caffe/layers/mkl_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
MKLEltwiseLayer<Dtype>::~MKLEltwiseLayer() {
  if (sumPrimitive != NULL)
    dnnDelete<Dtype>(sumPrimitive);
}

template <typename Dtype>
void MKLEltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0
      || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
      "MKLEltwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation()
      == EltwiseParameter_EltwiseOp_PROD
      && this->layer_param().eltwise_param().coeff_size())) <<
      "MKLEltwise layer only takes coefficients for summation.";

  CHECK(this->layer_param().eltwise_param().operation() ==
    EltwiseParameter_EltwiseOp_SUM)
      << "MKLEltwise Layer only process summation.";

  op_ = this->layer_param_.eltwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(bottom.size(), 1);
  if (this->layer_param().eltwise_param().coeff_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
    }
  }
  stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();

  num_bottoms = bottom.size();
  size_t dim_src = bottom[0]->shape().size();
  // size_t dim_dst = dim_src;

  dnnError_t e;

  size_t sizes_src[dim_src], strides_src[dim_src];
  for (size_t d = 0; d < dim_src; ++d) {
      sizes_src[d] = bottom[0]->shape()[dim_src - d - 1];
      strides_src[d] = (d == 0) ? 1 : strides_src[d-1]*sizes_src[d-1];
  }

  for (size_t i = 0; i < num_bottoms; ++i) {
      fwd_bottom_data.push_back(
        shared_ptr<MKLData<Dtype> >(new MKLData<Dtype>));
      CHECK_EQ(dim_src, bottom[i]->shape().size());
      e = dnnLayoutCreate<Dtype>(&(fwd_bottom_data[i]->layout_usr),
         dim_src, sizes_src, strides_src);
      CHECK_EQ(e, E_SUCCESS);
  }

  e = dnnLayoutCreate<Dtype>(&fwd_top_data->layout_usr,
    dim_src, sizes_src, strides_src);
  CHECK_EQ(e, E_SUCCESS);
}

template <typename Dtype>
void MKLEltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
  // If max operation, we will initialize the vector index part.
  if (this->layer_param_.eltwise_param().operation() ==
      EltwiseParameter_EltwiseOp_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->shape());
  }
}

template <typename Dtype>
void MKLEltwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  dnnError_t e;
  vector<void*> bottom_data;
  bool num_prv = 0;
  for (size_t i = 0; i < num_bottoms; i++) {
    bottom_data.push_back(
      reinterpret_cast<void *>(const_cast<Dtype*>(bottom[i]->prv_data())));
    if (bottom_data[i] != NULL) {
      num_prv += 1;
    } else {
      bottom_data[i] =
        reinterpret_cast<void *>(const_cast<Dtype*>(bottom[i]->cpu_data()));
    }
  }

  if (num_prv > 0) {
    if (sumPrimitive == NULL) {
      dnnLayout_t int_layout = NULL;
      for (size_t i = 0; i < num_bottoms; ++i) {
        if (bottom[i]->prv_data() != NULL) {
          CHECK((bottom[i]->get_prv_descriptor_data())->get_descr_type()
            == PrvMemDescr::PRV_DESCR_MKL2017);
          shared_ptr<MKLData<Dtype> > mem_descr =
              boost::static_pointer_cast<MKLData<Dtype> >(
                bottom[i]->get_prv_descriptor_data());
          CHECK(mem_descr != NULL);
          fwd_bottom_data[i] = mem_descr;
          if (int_layout == NULL) {
            int_layout = mem_descr->layout_int;
          }
        }
      }
      e = dnnSumCreate<Dtype>(&sumPrimitive, NULL,
        num_bottoms, int_layout, &coeffs_[0]);
      CHECK_EQ(e, E_SUCCESS);

      e = dnnLayoutCreateFromPrimitive<Dtype>(&fwd_top_data->layout_int,
        sumPrimitive, dnnResourceDst);
      CHECK_EQ(e, E_SUCCESS);
      fwd_top_data->create_conversions();

      for (int i = 0; i < num_bottoms; ++i) {
        if (bottom[i]->prv_data() == NULL) {
          e = dnnLayoutCreateFromPrimitive<Dtype>(
            &fwd_bottom_data[i]->layout_int, sumPrimitive,
              (dnnResourceType_t)(dnnResourceMultipleSrc + i));
          CHECK_EQ(e, E_SUCCESS);
          fwd_bottom_data[i]->create_conversions();
        }
      }
    }
  } else {
    if (sumPrimitive == NULL) {
      e = dnnSumCreate<Dtype>(&sumPrimitive, NULL, num_bottoms,
        fwd_top_data->layout_usr, &coeffs_[0]);
      CHECK_EQ(e, E_SUCCESS);
    }
  }

  switch (op_) {
  case EltwiseParameter_EltwiseOp_SUM:
    void *eltwise_res[dnnResourceNumber];
    for (int i = 0; i < num_bottoms; ++i) {
      if (fwd_bottom_data[i]->convert_to_int) {
        eltwise_res[dnnResourceMultipleSrc + i] =
          fwd_bottom_data[i]->get_converted_prv(bottom[i], false);
      } else {
        eltwise_res[dnnResourceMultipleSrc + i] =
          reinterpret_cast<void *>(bottom_data[i]);
      }
    }

    if (fwd_top_data->convert_from_int) {
      top[0]->set_prv_data(fwd_top_data->internal_ptr, fwd_top_data, false);
      eltwise_res[dnnResourceDst] =
        reinterpret_cast<void*>(const_cast<Dtype*>(fwd_top_data->internal_ptr));
    } else {
      eltwise_res[dnnResourceDst] =
        reinterpret_cast<void*>(const_cast<Dtype*>(top[0]->mutable_cpu_data()));
    }

    e = dnnExecute<Dtype>(sumPrimitive, eltwise_res);
    CHECK_EQ(e, E_SUCCESS);

    break;
  case EltwiseParameter_EltwiseOp_PROD:
  case EltwiseParameter_EltwiseOp_MAX:
    LOG(FATAL) << "Unsupported elementwise operation.";
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
void MKLEltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // const int count = top[0]->count();
  // const Dtype* top_data = top[0]->cpu_data();
  // const Dtype* top_diff = top[0]->cpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      // Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_SUM:
           CHECK_EQ(coeffs_[i], Dtype(1)) << "Not supported yet";        
           bottom[i]->ShareDiff(*top[0]);
        break;
      case EltwiseParameter_EltwiseOp_MAX:
      case EltwiseParameter_EltwiseOp_PROD:
        LOG(FATAL) << "Unsupported elementwise operation.";
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MKLEltwiseLayer);
#else
template <typename Dtype>
void MKLEltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {NOT_IMPLEMENTED;}
template <typename Dtype>
void MKLEltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {NOT_IMPLEMENTED;}
#endif

INSTANTIATE_CLASS(MKLEltwiseLayer);
}  // namespace caffe
#endif  // #if defined(MKL2017_SUPPORTED) && defined(USE_MKL2017_NEW_API)
