/**
*  BigDataLayer
* Two importatnt notes. This layer can return labels and/or IDS (row numbers). The reson for IDs
* is that it is possible to find out which row was missclassified and tune model settings.
*
*/
#include <stdint.h>
#include <cstdio> // rename(char*, char*)

#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <boost/thread.hpp>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
BigDataLayer<Dtype>::~BigDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  if (textstream_ != NULL) {
    if(textstream_->is_open()) textstream_->close();
    delete textstream_;
  }
  if (binstream_ != NULL) {
    if(binstream_->is_open()) binstream_->close();
    delete binstream_;
  }
}

template <typename Dtype>
void BigDataLayer<Dtype>::ResetStream(std::fstream* newstream) {
  this->JoinPrefetchThread();
  LOG(WARNING) << "BigData stream reset called";
  if(textstream_ != NULL) delete textstream_;
  textstream_ = newstream;
  this->CreatePrefetchThread();
  this->cur_id_ = 0;
}

template <typename Dtype>
void BigDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CPUTimer init_timer;
  init_timer.Start();
  const BigDataParameter& big_params = this->layer_param_.big_data_param();
  file_smaller_than_chunk_ = false;
  already_loaded_ = false;
  cur_id_ = 0;

  // check the right count of blobs
  this->output_labels_ = top.size() > 1;
  has_label_ = (big_params.has_label() && big_params.label() != -1);
  has_ids_ = ((!has_label_ && top.size() == 2) || (top.size() == 3));

  // save metainfo about textual parameters
  delim_ = big_params.separator().c_str()[0];
  newline_ = big_params.newline().c_str()[0];

  // save indices describing data
  data_start_ = big_params.data_start();
  data_end_ = big_params.data_end();
  label_ = big_params.label();
  size_t data_cols = data_end_ - data_start_ + 1 + (has_label_ ? 1 : 0);

  // first check if we have already a binary-csv
  switch (big_params.cache())
  {
    case ::caffe::BigDataParameter_CacheControl_ENABLED:
      binstream_ = new std::fstream(string(big_params.source() + "bin").c_str(), 
                                    std::ios::in | std::ios::binary);
      has_binary_ = binstream_->good();
      cache_ = true;
      break;
    case ::caffe::BigDataParameter_CacheControl_RENEW:
      cache_ = true;
      has_binary_ = false;
      break;
    case ::caffe::BigDataParameter_CacheControl_DISABLED:
      cache_ = false;
      has_binary_ = false;
      break;
  }

  if(cache_ && binstream_->good())
  {
    // find out if the data will fit in one batch (so we can cache them)
    binstream_->seekg(0, std::ios_base::end);
    file_smaller_than_chunk_ = (big_params.chunk_size() * 1000000) > (binstream_->tellg());
    binstream_->seekg(0);
    LOG(INFO) << "Using converted BINARY CSV file instead the original one!";
  }
  else
  {
    // open file for reading and in case of IO problems throw an exception
    textstream_ = new std::fstream(
      big_params.source().c_str(), std::ios::in);
    if (!textstream_->good()) throw std::ifstream::failure(
      string("cannot open for reading file ") +
      big_params.source());

    // find out if the data will fit in one batch (so we can cache them)
    // meanwhile check if we have enough delimiters to claimed data columns
    char *buff = new char[21*data_cols + 1];
    textstream_->getline(buff, 21*data_cols);

    // count number of delimiters in one line
    int count = 0;
    char* p = buff;
    while(*(p++) != '\0') if(*p == delim_) ++count;
    if(count < (data_cols-1)) {
      LOG(ERROR) << "Found only " << count << " delimiters '" << delim_ 
                 << "' in line " << buff << std::endl;
      throw std::ifstream::failure("Not enough data columns in source file?");
    }
    delete[] buff;
    p = NULL;

    // estimate the size of file
    textstream_->seekg(0, std::ios_base::end);
    file_smaller_than_chunk_ =
      (big_params.chunk_size() * 1000000) > (textstream_->tellg()/2);
    textstream_->seekg(0);


    // skip # of lines denoted to header
    for(int i=0; i < big_params.header() + int(rand() / RAND_MAX * big_params.rand_skip()); ++i) {
      textstream_->ignore(2147483647, newline_);
    }

    // init binary stream for writing
    if(cache_) {
      delete binstream_;
      binstream_ = new std::fstream(string(big_params.source()+"bin.part").c_str(),
                                    std::ios::out | std::ios::binary);
    }
  }

  if (file_smaller_than_chunk_) {
    LOG(INFO) << "Source file is LESS than one chunk - will use memory caching";
  }

  shape_ = vector<int>(4);
  // if the user has specified shape that means that his data are multi-dimensional
  // if(big_params.has_shape()) {
  //   shape_[3] = big_params.shape().width();
  //   shape_[2] = big_params.shape().height();
  //   shape_[1] = big_params.shape().channel();
  // } else {
    // otherwise we axpect one dimensional data
    shape_[3] = data_end_ - data_start_ + 1;
    shape_[2] = 1;
    shape_[1] = 1;
  // }
  // the maximal batch size is computed from chunk_size (which is in MB)
  shape_[0] = ceil((1000000 / (sizeof(Dtype) * shape_[3] * shape_[2] * shape_[1])) *
                   big_params.chunk_size());
  top[0]->Reshape(shape_);
  this->prefetch_data_.Reshape(shape_);

  // Read a data point, and use it to initialize the top blob.
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // every other blob will be of size BATCH, 1, 0, 0
  label_shape_ = vector<int>(1, shape_[0]);
  for(int i=1; i < top.size(); ++i) {
    top[i]->Reshape(label_shape_);
  }
  this->prefetch_label_.Reshape(label_shape_);
  if(has_ids_) this->prefetch_ids_.Reshape(label_shape_);

  init_timer.Stop();
  DLOG(INFO) << "Init BigDataLayer time: " << init_timer.MilliSeconds() << " ms.";
}


template <typename Dtype>
void BigDataLayer<Dtype>::ReadFromText(size_t how_many, Dtype* data, Dtype* labels, Dtype* ids)
{
  const BigDataParameter& big_params = this->layer_param_.big_data_param();
  char buff[255];
  size_t data_total = (shape_[1] * shape_[2] * shape_[3]);
  size_t batch = 0, current_col = 0, data_col = 0;
  bool got_label = false;

  while(batch < how_many)
  {
    data_col = 0;
    current_col = 0;
    got_label = false;
    // read a row from text file
    while( textstream_->good() && (has_label_ != got_label || data_col < data_total) )
    {
      textstream_->getline(buff, 255, delim_);
      if(current_col >= data_start_ && current_col <= data_end_) {
        data[data_col] = atof(buff);
        ++data_col;
      }

      if(current_col == label_) {
        *labels = atof(buff);
        got_label = true;
      }

      ++current_col;
    }
    textstream_->ignore(2147483647, newline_);

    if (data_col == data_total) {
      // save the row to the binary file
      if(cache_) {
        if(has_label_) binstream_->write((char*)labels, sizeof(*labels));
        binstream_->write((char*) data, data_total * sizeof(*data));
      }
      // move the pointers
      data += data_total;
      if(has_label_) labels += 1;
      if(ids != NULL) ids[batch] = cur_id_;
      ++batch;
      ++cur_id_;
      // LOG(INFO) << " CSV " << batch << "/" << how_many << std::setw(3)
      //           << "\tD: " << data[-data_total] << ", " << data[-data_total+1] 
      //                     << "..." << data[-2] << ", " << data[-1]
      //           << "\tL: " << labels[-1] // TODO: dangerous, remove
      //           << "\tI: " << ids[batch];
    }

    // if we reached EOF || there was an empty line at the end of the file
    if(textstream_->eof() || data_col < data_total) {
      // LOG(INFO) << "Reset file at batch " << batch;
      textstream_->close();
      delete textstream_;
      textstream_ = NULL;
      cur_id_ = 0;
      if(cache_)
      {
        binstream_->close();
        std::rename( // strip the ".part" extension when the file is done
          string(big_params.source()+"bin.part").c_str(),
          string(big_params.source()+"bin").c_str());
        // reuse the binary file
        delete binstream_;
        binstream_ = new std::fstream(
          string(big_params.source() + "bin").c_str(),
          std::ios::in | std::ios::binary);
        has_binary_ = true;
        // means there was an empty line at the end of the file, so restart loading
        return ReadFromBin(how_many - batch, data, labels, ids);
      } 
      else
      {
        textstream_ = new std::fstream(
          big_params.source().c_str(), std::ios::in);
        // skip # of lines denoted to header
        for(int i=0; i < big_params.header() + int(rand() / RAND_MAX * big_params.rand_skip()); ++i) {
          textstream_->ignore(2147483647, newline_);
        }
      }
    }
  }
}


template <typename Dtype>
void BigDataLayer<Dtype>::ReadFromBin(size_t how_many, Dtype* data, Dtype* labels, Dtype *ids)
{
  // LOG(INFO) << " BIN,  STARTS on pos: " << binstream_->tellg();
  size_t data_total = (shape_[1] * shape_[2] * shape_[3]);

  for(int batch=0; batch < how_many; ++batch)
  {
    if(ids != NULL) ids[batch] = cur_id_;
    ++cur_id_;
    if(has_label_) {
      binstream_->read((char*)labels, sizeof(*labels));
      labels += 1;
    }
    binstream_->read((char*)data, data_total * sizeof(*data));
    data += data_total;
    if(binstream_->eof()) {
      // this flags get set in case of unsucessful read -- means that we should
      // check it always after the first read and then re-read ... here we leave
      // one old datum in the prefetched_data ... which doesn't hurt, does it?
      binstream_->close();
      delete binstream_;
      binstream_ = new std::fstream(string(this->layer_param_.big_data_param().source() + "bin").c_str(),
        std::ios::in | std::ios::binary);
      cur_id_ = 0;
      // skip random number of lines between 0 and BigDataParams.rand_skip
      binstream_->seekg((data_total * sizeof(*data) + sizeof(*labels)) * 
                        int(rand()/RAND_MAX * this->layer_param_.big_data_param().rand_skip()));
    }
    // LOG(INFO) << " BIN " << batch << "/" << how_many
    // << " D: " << data[-data_total] << "..." << data[-1]
    // << " L: " << labels[-1]; // TODO: dangerous, remove
  }
}


// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void BigDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();

  // if we have read the whole file and it's smaller than chunk, don't read it again
  if(file_smaller_than_chunk_ && already_loaded_) {
    batch_timer.Stop();
    // LOG(INFO) << "File smaller than batch -- using last loaded data";
    return;
  }

  // revert prefetched data and labels back to it's full size
  this->prefetch_data_.Reshape(shape_);
  if(has_label_) this->prefetch_label_.Reshape(label_shape_);

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = NULL;
  Dtype* top_ids = NULL;

  if(has_label_) {
    top_label = this->prefetch_label_.mutable_cpu_data();
    if(has_ids_) top_ids = this->prefetch_ids_.mutable_cpu_data();
  } else if (has_ids_) {
    top_ids = this->prefetch_label_.mutable_cpu_data();
  }
  if(has_binary_)
    ReadFromBin(shape_[0], top_data, top_label, top_ids);
  else
    ReadFromText(shape_[0], top_data, top_label, top_ids);

  already_loaded_ = true;

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

template <typename Dtype>
void BigDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  this->JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  // if there are more "tops" fill them from appropriate sources
  if (top.size() > 1) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());
  }
  if(top.size() > 2) {
    caffe_copy(this->prefetch_ids_.count(), this->prefetch_ids_.cpu_data(),
               top[2]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(BigDataLayer);
REGISTER_LAYER_CLASS(BigData);

}  // namespace caffe
