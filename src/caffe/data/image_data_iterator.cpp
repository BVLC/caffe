// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>

#include "caffe/data/data.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
using std::string;
using std::vector;

template<typename Dtype>
ImageDataIterator<Dtype>::ImageDataIterator(const DataIteratorParameter& param):
	 DataIterator<Dtype>(param), base_path_("") {
	if (this->data_iterator_param_.image_data_param().has_base_path()) {
		base_path_ = this->data_iterator_param_.image_data_param().base_path();
	}
	if (base_path_.rfind("/") != (base_path_.size() - 1)) {
		base_path_ += "/";
	}
}

template<typename Dtype>
void ImageDataIterator<Dtype>::Init() {
	// Read the file with filenames and labels
	const string& source =
			this->data_iterator_param_.image_data_param().source();
	LOG(INFO) << "Opening file " << source;
	std::ifstream infile(source.c_str());
	string filename;
	int label;
	while (infile >> filename >> label) {
		lines_.push_back(std::make_pair(filename, label));
	}
	LOG(INFO) << "A total of " << lines_.size() << " images.";
	Shuffle();
}

template<typename Dtype>
bool ImageDataIterator<Dtype>::HasNext() {
}

template<typename Dtype>
const DataBatch<Dtype>& ImageDataIterator<Dtype>::Next() const {
}

template<typename Dtype>
void ImageDataIterator<Dtype>::Shuffle() {
	if (this->data_iterator_param_.image_data_param().shuffle()) {
		// randomly shuffle data
		LOG(INFO) << "Shuffling data";
		std::random_shuffle(lines_.begin(), lines_.end());
	}
}

INSTANTIATE_CLASS(ImageDataIterator);

}  // namespace caffe
