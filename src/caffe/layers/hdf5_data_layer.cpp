#ifdef USE_HDF5
/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layers/hdf5_data_layer.hpp"
#include "caffe/util/hdf5.hpp"

namespace std {
  template<class Dtype> struct hash<
    std::unique_ptr<caffe::hdf5DataLayerDetail::HDF5FileDataHandler<Dtype>>> {
    typedef std::unique_ptr<
      caffe::hdf5DataLayerDetail::HDF5FileDataHandler<Dtype>> argument_type;
    typedef std::size_t result_type;
    result_type operator()(argument_type const& s) const noexcept {
      result_type h = 0;
      for (const auto& a : s->files())
          h ^= (std::hash<std::string> {}(a) << 1);
      return h;
    }
  };

  template<class Dtype> struct equal_to<
    std::unique_ptr<caffe::hdf5DataLayerDetail::HDF5FileDataHandler<Dtype>>> {
    typedef std::unique_ptr<
      caffe::hdf5DataLayerDetail::HDF5FileDataHandler<Dtype>>
        first_argument_type;
    typedef first_argument_type second_argument_type;
    typedef bool result_type;

    result_type operator()(first_argument_type const& a,
        second_argument_type const& b) const noexcept {
      return std::equal_to<std::vector<std::string>> {}(a->files(), b->files());
    }
  };
}  // namespace std

namespace caffe {

namespace hdf5DataLayerDetail {

template <typename Dtype>
HDF5FileDataHandler<Dtype>::HDF5FileDataHandler(
    const std::vector<std::string>& files, const LayerParameter& layer_param)
  : hdf_filenames_(files), file_permutation_(files.size()),
    layer_param_(layer_param) {
  // Default to identity permutation.
  for (int i = 0; i < file_permutation_.size(); i++) {
    file_permutation_[i] = i;
  }

  // Shuffle if needed.
  if (this->layer_param_.hdf5_data_param().shuffle()) {
    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
  }
}

template <typename Dtype>
void HDF5DataManager<Dtype>::createInstance() {
  std::unique_lock<std::mutex> lock(_createInstanceMutex, std::try_to_lock);
  if (lock) {
    _instance.reset(new HDF5DataManager);
  }
}

template <typename Dtype>
HDF5FileDataHandler<Dtype>* HDF5DataManager<Dtype>::registerFileSet
  (const std::vector<std::string>& files, const LayerParameter& layer_param) {
  auto handler = std::unique_ptr<HDF5FileDataHandler<Dtype>>(
      new HDF5FileDataHandler<Dtype>(files, layer_param));
  std::unique_lock<std::mutex> lock(instanceMutex_);
  auto ins = handlers_.insert(std::move(handler));
  // we do not care if insertion was blocked,
  // if it was we use the present instance
  return ins.first->get();
}

template <typename Dtype>
std::shared_ptr<HDF5FileDataBuffer<Dtype>> HDF5FileDataHandler<Dtype>::getBuffer
  (std::shared_ptr<HDF5FileDataBuffer<Dtype>> prev) {
  if (!current_buffer_.expired()) {
    auto tmp = current_buffer_.lock();
    if (prev != tmp)
      return tmp;
  }
  std::unique_lock<std::mutex> loadDataLock(loadDataMutex_, std::try_to_lock);
  if (loadDataLock) {
    ++current_file_;
    if (current_file_ == file_permutation_.size()) {
      current_file_ = 0;
      DLOG(INFO) << "Looping around to first file.";
    }
    auto tmp = std::make_shared<HDF5FileDataBuffer<Dtype>>(current_file_,
        hdf_filenames_[file_permutation_[current_file_]], layer_param_);
    current_buffer_ = tmp;
    return tmp;
  } else {
    loadDataLock.lock();
    return current_buffer_.lock();
  }
}

// Load data and label from HDF5 filename into the class property blobs.
// template <typename Dtype>
// void HDF5DataLayer<Dtype>::LoadHDF5FileData(const char* filename) {
template <typename Dtype>
HDF5FileDataBuffer<Dtype>::HDF5FileDataBuffer
  (unsigned int idx, const std::string& filename
    , const LayerParameter& layer_param) {
  DLOG(INFO) << "Loading HDF5 file: " << filename;

  hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  int top_size = layer_param.top_size();
  hdf_blobs_.resize(top_size);

  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;

  for (int i = 0; i < top_size; ++i) {
    hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    // Allow reshape here, as we are loading data not params
    hdf5_load_nd_dataset(file_id, layer_param.top(i).c_str(),
        MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get(), true);
  }

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

  // MinTopBlobs==1 guarantees at least one top blob
  CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
  const int num = hdf_blobs_[0]->shape(0);
  for (int i = 1; i < top_size; ++i) {
    CHECK_EQ(hdf_blobs_[i]->shape(0), num);
  }
  // Default to identity permutation.
  data_permutation_.resize(hdf_blobs_[0]->shape(0));
  for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
    data_permutation_[i] = i;

  // Shuffle if needed.
  if (layer_param.hdf5_data_param().shuffle()) {
    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    DLOG(INFO) << "Successfully loaded " << hdf_blobs_[0]->shape(0)
               << " rows (shuffled)";
  } else {
    DLOG(INFO) << "Successfully loaded " << hdf_blobs_[0]->shape(0) << " rows";
  }
}

}  // namespace hdf5DataLayerDetail

template <typename Dtype>
HDF5DataLayer<Dtype>::~HDF5DataLayer<Dtype>() { }

template <typename Dtype>
void HDF5DataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Refuse transformation parameters since HDF5 is totally generic.
  CHECK(!this->layer_param_.has_transform_param()) <<
      this->type() << " does not transform data.";
  // Read the source to parse the filenames.
  const string& source = this->layer_param_.hdf5_data_param().source();
  LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
  std::vector<std::string> hdf_filenames;
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf_filenames.push_back(line);
    }
  } else {
    LOG(FATAL) << "Failed to open source file: " << source;
  }
  source_file.close();
  LOG(INFO) << "Number of HDF5 files: " << hdf_filenames.size();
  CHECK_GE(hdf_filenames.size(), 1)
    << "Must have at least 1 HDF5 filename listed in " << source;

  const int top_size = this->layer_param_.top_size();
  data_handler_ =
    hdf5DataLayerDetail::HDF5DataManager<Dtype>::instance().registerFileSet
    (hdf_filenames, this->layer_param_);


  current_buffer_ = data_handler_->getBuffer(nullptr);
  current_row_ = 0;

  // Reshape blobs.
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  vector<int> top_shape;
  for (int i = 0; i < top_size; ++i) {
    top_shape.resize(current_buffer_->hdf_blobs()[i]->num_axes());
    top_shape[0] = batch_size;
    for (int j = 1; j < top_shape.size(); ++j) {
      top_shape[j] = current_buffer_->hdf_blobs()[i]->shape(j);
    }
    top[i]->Reshape(top_shape);
  }
}

template <typename Dtype>
bool HDF5DataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void HDF5DataLayer<Dtype>::Next() {
  if (++current_row_ == current_buffer_->hdf_blobs()[0]->shape(0)) {
    current_buffer_ = data_handler_->getBuffer(current_buffer_);
    current_row_ = 0;
  }
  offset_++;
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  for (int i = 0; i < batch_size; ++i) {
    while (Skip()) {
      Next();
    }
    for (int j = 0; j < this->layer_param_.top_size(); ++j) {
      int data_dim = top[j]->count() / top[j]->shape(0);
      caffe_copy(data_dim,
          &current_buffer_->hdf_blobs()[j]->cpu_data()[
            current_buffer_->data_permutation()[current_row_] * data_dim],
          &top[j]->mutable_cpu_data()[i * data_dim]);
    }
    Next();
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDF5DataLayer, Forward);
#endif

INSTANTIATE_CLASS(HDF5DataLayer);
REGISTER_LAYER_CLASS(HDF5Data);

}  // namespace caffe
#endif  // USE_HDF5
