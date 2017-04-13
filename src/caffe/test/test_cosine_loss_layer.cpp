#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/cosine_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CosineLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CosineLossLayerTest()
     : blob_bottom_data_(new Blob<Dtype>(100, 50, 1, 1)),
       blob_bottom_labels_(new Blob<Dtype>(100, 50, 1, 1)),
       blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_labels_);
    blob_bottom_vec_.push_back(blob_bottom_labels_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~CosineLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_labels_;
    delete blob_top_loss_;
  }

  int offset(int n, int c, int h, int w, const vector<int>& shape_vec) {
      return ((n * shape_vec[1] + c)
              * shape_vec[2] + h)
              * shape_vec[3] + w;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_labels_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CosineLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(CosineLossLayerTest, TestForward2D) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CosineLossLayer<Dtype> layer(layer_param);
  vector<int> shape_vec;
  shape_vec.push_back(1);
  shape_vec.push_back(2);
  Blob<Dtype>* blob_inp = new Blob<Dtype>(shape_vec);
  Blob<Dtype>* blob_label = new Blob<Dtype>(shape_vec);
  vector<Blob<Dtype>*> bottom_vec;
  bottom_vec.push_back(blob_inp);
  bottom_vec.push_back(blob_label);
  layer.LayerSetUp(bottom_vec, this->blob_top_vec_);

  blob_inp->mutable_cpu_data()[0] = Dtype(1);
  blob_inp->mutable_cpu_data()[1] = Dtype(0);
  blob_label->mutable_cpu_data()[0] = Dtype(0);
  blob_label->mutable_cpu_data()[1] = Dtype(1);
  layer.Forward(bottom_vec, this->blob_top_vec_);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], Dtype(1), 0.00001);

  blob_inp->mutable_cpu_data()[0] = Dtype(0.35);
  blob_inp->mutable_cpu_data()[1] = Dtype(0.18);
  blob_label->mutable_cpu_data()[0] = Dtype(0.47);
  blob_label->mutable_cpu_data()[1] = Dtype(0.59);
  layer.Forward(bottom_vec, this->blob_top_vec_);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], Dtype(0.088185669), 0.00001);

  bottom_vec.clear();
  shape_vec.clear();
  delete blob_inp;
  delete blob_label;
}

TYPED_TEST(CosineLossLayerTest, TestForwardND) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CosineLossLayer<Dtype> layer(layer_param);
  vector<int> shape_vec;
  shape_vec.push_back(1);
  shape_vec.push_back(3);
  shape_vec.push_back(2);
  shape_vec.push_back(2);
  Blob<Dtype>* blob_inp = new Blob<Dtype>(shape_vec);
  Blob<Dtype>* blob_label = new Blob<Dtype>(shape_vec);
  vector<Blob<Dtype>*> bottom_vec;
  bottom_vec.push_back(blob_inp);
  bottom_vec.push_back(blob_label);
  layer.LayerSetUp(bottom_vec, this->blob_top_vec_);

  Dtype* inp_data = blob_inp->mutable_cpu_data();
  Dtype* label_data = blob_label->mutable_cpu_data();
  inp_data[this->offset(0, 0, 0, 0, shape_vec)] = Dtype(0.83);
  inp_data[this->offset(0, 1, 0, 0, shape_vec)] = Dtype(0.41);
  inp_data[this->offset(0, 2, 0, 0, shape_vec)] = Dtype(0.5);
  inp_data[this->offset(0, 0, 0, 1, shape_vec)] = Dtype(0.19);
  inp_data[this->offset(0, 1, 0, 1, shape_vec)] = Dtype(0.73);
  inp_data[this->offset(0, 2, 0, 1, shape_vec)] = Dtype(0.0);
  inp_data[this->offset(0, 0, 1, 0, shape_vec)] = Dtype(0.3);
  inp_data[this->offset(0, 1, 1, 0, shape_vec)] = Dtype(0.52);
  inp_data[this->offset(0, 2, 1, 0, shape_vec)] = Dtype(0.8);
  inp_data[this->offset(0, 0, 1, 1, shape_vec)] = Dtype(0.33);
  inp_data[this->offset(0, 1, 1, 1, shape_vec)] = Dtype(0.91);
  inp_data[this->offset(0, 2, 1, 1, shape_vec)] = Dtype(0.03);

  label_data[this->offset(0, 0, 0, 0, shape_vec)] = Dtype(0.91);
  label_data[this->offset(0, 1, 0, 0, shape_vec)] = Dtype(0.48);
  label_data[this->offset(0, 2, 0, 0, shape_vec)] = Dtype(0.97);
  label_data[this->offset(0, 0, 0, 1, shape_vec)] = Dtype(0.32);
  label_data[this->offset(0, 1, 0, 1, shape_vec)] = Dtype(0.88);
  label_data[this->offset(0, 2, 0, 1, shape_vec)] = Dtype(0.21);
  label_data[this->offset(0, 0, 1, 0, shape_vec)] = Dtype(0.74);
  label_data[this->offset(0, 1, 1, 0, shape_vec)] = Dtype(0.89);
  label_data[this->offset(0, 2, 1, 0, shape_vec)] = Dtype(0.63);
  label_data[this->offset(0, 0, 1, 1, shape_vec)] = Dtype(0.88);
  label_data[this->offset(0, 1, 1, 1, shape_vec)] = Dtype(0.01);
  label_data[this->offset(0, 2, 1, 1, shape_vec)] = Dtype(0.65);

  layer.Reshape(bottom_vec, this->blob_top_vec_);
  layer.Forward(bottom_vec, this->blob_top_vec_);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], Dtype(0.21489816), 0.00001);

  bottom_vec.clear();
  shape_vec.clear();
  delete blob_inp;
  delete blob_label;
}

TYPED_TEST(CosineLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CosineLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_, 0);
}

}  // namespace caffe
