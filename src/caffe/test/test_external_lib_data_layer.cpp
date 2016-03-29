#include <string>
#include <vector>

#include "caffe/layers/external_lib_data_layer.hpp"
#include "caffe/util/external_lib_data_source.hpp"
#include "caffe/util/external_lib_data_source_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

// Names of the top blobs.
const char* top_input_name = "input";
const char* top_label_name = "label";

// Dimension properties of the top blobs.
static const int batch_size = 2;
static const int input_dims = 3;
static const int channels = 3;
static const int height = 30;
static const int width = 40;
static const int input_count = channels * height * width;
static const int target_dims = 1;

// Dummy IDatum implementation. Sets same (given) value for input and target.
template <typename Dtype>
class DatumTest : public IDatum {
 public:
  explicit DatumTest(Dtype value) {
    target_ = value;

    input_ = shared_ptr<Dtype[]>(new Dtype[input_count]);
    for (int i = 0; i < input_count; i++) {
      input_[i] = value;
    }
  }

  virtual void GetBlobShape(const char* blob_name, const int** shape,
    int* shape_count) {
    if (strcmp(blob_name, top_input_name) == 0) {
      *shape = input_shape;
      *shape_count = input_dims;
    } else if (strcmp(blob_name, top_label_name) == 0) {
      *shape = &target_shape_;
      *shape_count = target_dims;
    } else {
      LOG(FATAL) << "Invalid blob name";
    }
  }

  virtual void GetBlobData(const char* blob_name, void* blob_data,
    BlobType type) {
    if (strcmp(blob_name, top_input_name) == 0) {
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      memcpy(blob_data, input_.get(), input_count * sizeof(Dtype));
    } else if (strcmp(blob_name, top_label_name) == 0) {
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      memcpy(blob_data, &target_, sizeof(Dtype));
    } else {
      LOG(FATAL) << "Invalid blob name";
    }
  }

 private:
  static const int input_shape[input_dims];
  boost::shared_ptr<Dtype[]> input_;

  static const int target_shape_;
  Dtype target_;
};

template <typename Dtype>
const int DatumTest<Dtype>::input_shape[input_dims] = { channels, height,
  width };

template <typename Dtype>
const int DatumTest<Dtype>::target_shape_ = 1;

// Dummy IExternalLibDataSource implementation. Creates two examples and
// cycles through them.
template <typename Dtype>
class ExternalLibDataSourceTest : public IExternalLibDataSource {
 public:
  ExternalLibDataSourceTest() : datum_1_(Dtype(1)), datum_2_(Dtype(2)),
    curr_datum_(&datum_1_) {}

  virtual ~ExternalLibDataSourceTest() {}

  virtual void Release() {
    delete this;
  }

  virtual void MoveToNext() {
    if (curr_datum_ == &datum_1_) {
      curr_datum_ = &datum_2_;
    } else if (curr_datum_ == &datum_2_) {
      curr_datum_ = &datum_1_;
    } else {
      LOG(FATAL) << "Invalid datum test value";
    }
  }

  virtual IDatum* GetCurrent() {
    return curr_datum_;
  }

 private:
  DatumTest<Dtype> datum_1_;
  DatumTest<Dtype> datum_2_;
  DatumTest<Dtype>* curr_datum_;
};

// Dummy implementation of the external lib. Does not load any external library
// but fakes implementation in memory.
template <typename Dtype>
class ExternalLibTest : public IExternalLib {
 public:
  virtual IExternalLibDataSource* GetDataSource() {
    return new ExternalLibDataSourceTest<Dtype>();
  }
};

// Extends original external lib data layer to override external lib getting
// (and to provide test implementation instead of original one).
template <typename Dtype>
class ExternalLibDataLayerEx : public ExternalLibDataLayer<Dtype> {
 public:
  explicit ExternalLibDataLayerEx(const LayerParameter& param) :
    ExternalLibDataLayer<Dtype>(param) {}

  // Override get library function.
  virtual boost::shared_ptr<IExternalLib> GetExternalLib(
    const std::string& ext_lib_path,
    const std::string& factory_name,
    const std::string& ext_lib_param) {
    return shared_ptr<IExternalLib>(new ExternalLibTest<Dtype>());
  }
};

// Test object.
template <typename TypeParam>
class ExternalLibDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ExternalLibDataLayerTest() {
    blob_top_vec_.push_back(&blob_top_data_);
    blob_top_vec_.push_back(&blob_top_label_);
  }

  virtual ~ExternalLibDataLayerTest() {}

  Blob<Dtype> blob_top_data_;
  Blob<Dtype> blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ExternalLibDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(ExternalLibDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;

  // Create layer parameters.
  LayerParameter param;
  param.add_top(top_input_name);
  param.add_top(top_label_name);
  ExternalLibDataParameter* ext_lib_data_param =
    param.mutable_external_lib_data_param();
  ext_lib_data_param->set_batch_size(batch_size);

  // Create layer based on layer parameters.
  ExternalLibDataLayerEx<Dtype> layer(param);

  // Test layer setup.
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Make sure we reshaped input accordingly.
  ASSERT_EQ(this->blob_top_data_.num(), batch_size);
  ASSERT_EQ(this->blob_top_data_.channels(), channels);
  ASSERT_EQ(this->blob_top_data_.height(), height);
  ASSERT_EQ(this->blob_top_data_.width(), width);
  // Make sure we reshaped target accordingly.
  ASSERT_EQ(this->blob_top_label_.num(), batch_size);
  ASSERT_EQ(this->blob_top_label_.channels(), 1);
  ASSERT_EQ(this->blob_top_label_.height(), 1);
  ASSERT_EQ(this->blob_top_label_.width(), 1);

  // Test forward pass.
  const int iter_count = 2;
  for (int i = 0; i < iter_count; i++) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Make sure we have expected values (all input equal to target).
    const Dtype* input_data = this->blob_top_data_.cpu_data();
    const Dtype* label_data = this->blob_top_label_.cpu_data();
    int count = this->blob_top_data_.count(1);
    for (int ib = 0; ib < batch_size; ib++) {
      int data_offset = this->blob_top_data_.offset(ib);
      int label_offset = this->blob_top_label_.offset(ib);

      for (int ic = 0; ic < count; ic++) {
        ASSERT_EQ(input_data[data_offset + ic], label_data[label_offset]);
      }
    }
  }
}

}  // namespace caffe
