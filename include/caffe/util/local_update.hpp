// Copyright 2014 BVLC and contributors.

#ifndef _CAFFE_UTIL_LOCAL_UPDATE_HPP_
#define _CAFFE_UTIL_LOCAL_UPDATE_HPP_

namespace caffe {

template <typename Dtype>
void local_update1_cpu(const Dtype* data_A, const Dtype* data_B, Dtype* data_R, const int filter_num,
	    const int location_num, const int output_num);

template <typename Dtype>
void local_update1_gpu(const Dtype* data_A, const Dtype* data_B, Dtype* data_R, const int filter_num,
	    const int location_num, const int output_num);

template <typename Dtype>
void local_update2_cpu(const Dtype* data_A, const Dtype* data_B, Dtype* data_R, const int filter_num,
	    const int location_num, const int output_num);

template <typename Dtype>
void local_update2_gpu(const Dtype* data_A, const Dtype* data_B, Dtype* data_R, const int filter_num,
	    const int location_num, const int output_num);

}  // namespace caffe

#endif  // _CAFFE_UTIL_LOCAL_UPDATE_HPP_
