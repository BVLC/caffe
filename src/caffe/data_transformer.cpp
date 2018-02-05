/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/im_transforms.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase), data_reader_used(NULL) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
  if (param_.has_resize_param()) {
    CHECK_GT(param_.resize_param().height(), 0);
    CHECK_GT(param_.resize_param().width(), 0);
  }
  if (param_.has_expand_param()) {
    CHECK_GT(param_.expand_param().max_expand_ratio(), 1.);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum, Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && rand_num_(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;
  const bool flow = param_.flow();

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = rand_num_(datum_height - crop_size + 1);
      w_off = rand_num_(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
          if (flow && c == 2 && do_mirror) {
            datum_element = 255-datum_element;
          }
        } else {
          datum_element = datum.float_data(data_index);
          if (flow && c == 2 && do_mirror) {
            datum_element = 255-datum_element;
          }
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum, Dtype* transformed_data,
                                       NormalizedBBox* crop_bbox, RandNumbers& rand_num,
                                       const bool do_mirror, const bool has_uint8,
                                       const bool has_mean_file, const bool has_mean_values)
{
  int transform_func_id = (do_mirror << 2) +
                          (has_mean_file << 1) +
                          has_mean_values;

  if (!has_uint8) {
    switch (transform_func_id) {
        case 0: Transform<false, false, false, false>(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 1: Transform<false, false, false, true >(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 2: Transform<false, false, true , false>(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 3: Transform<false, false, true , true >(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 4: Transform<false, true , false, false>(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 5: Transform<false, true , false, true >(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 6: Transform<false, true , true , false>(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 7: Transform<false, true , true , true >(datum, transformed_data,
          crop_bbox, rand_num); break;
    }
  } else {
    switch (transform_func_id) {
        case 0: Transform<true, false, false, false>(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 1: Transform<true, false, false, true >(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 2: Transform<true, false, true , false>(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 3: Transform<true, false, true , true >(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 4: Transform<true, true , false, false>(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 5: Transform<true, true , false, true >(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 6: Transform<true, true , true , false>(datum, transformed_data,
          crop_bbox, rand_num); break;
        case 7: Transform<true, true , true , true >(datum, transformed_data,
          crop_bbox, rand_num); break;
    }
  }
}

namespace {
  // Based on the path we're in (detection or classification), perform transformations on
  // annotations.
  template<typename AnnotationHandler>
  void call_annotation_handler(AnnotationHandler& anno_handler, const bool do_resize, const bool do_mirror)
  {
    anno_handler(do_resize, do_mirror);
  }

  template<>
  void call_annotation_handler<EmptyType>(EmptyType&, const bool, const bool)
  {
  }
}

template<typename Dtype>
template<typename AnnotationHandler>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data,
                                       NormalizedBBox* crop_bbox,
                                       RandNumbers& rand_num,
                                       AnnotationHandler anno_handler)
{
  const bool do_mirror = param_.mirror() && rand_num(2);
  const string& data = datum.data();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  Transform(datum, transformed_data, crop_bbox, rand_num,
            do_mirror, has_uint8, has_mean_file, has_mean_values);

  call_annotation_handler(anno_handler, /* do_resize */ true, do_mirror);
}

template<typename Dtype>
template<bool has_uint8, bool do_mirror, bool has_mean_file,
  bool has_mean_values>
void DataTransformer<Dtype>::Transform(const Datum& datum_in,
                                       Dtype* transformed_data,
                                       NormalizedBBox* crop_bbox,
                                       RandNumbers& rand_num) {
  const Datum *datum = &datum_in;
  Datum resized_datum;
  if (param_.has_random_resize_param()) {
#ifdef USE_OPENCV
    RandomResizeImage(datum_in, &resized_datum);
    datum = &resized_datum;
#else
    LOG(FATAL) << "Random image resizing requires OpenCV; compile with USE_OPENCV.";
#endif
  } else if (param_.has_random_aspect_ratio_param()) {
#ifdef USE_OPENCV
    RandomAlterAspectRatio(datum_in, &resized_datum);
    datum = &resized_datum;
#else
    LOG(FATAL) << "Aspect ratio changes require OpenCV; compile with USE_OPENCV.";
#endif
  }
  const string& data = datum->data();
  const int datum_channels = datum->channels();
  const int datum_height = datum->height();
  const int datum_width = datum->width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = rand_num(datum_height - crop_size + 1);
      w_off = rand_num(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  // Return the normalized crop bbox.
  crop_bbox->set_xmin(Dtype(w_off) / datum_width);
  crop_bbox->set_ymin(Dtype(h_off) / datum_height);
  crop_bbox->set_xmax(Dtype(w_off + width) / datum_width);
  crop_bbox->set_ymax(Dtype(h_off + height) / datum_height);

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum->float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::GenerateRandNumbers(PreclcRandomNumbers& rn, bool sample_bboxes) {
  int count = (sample_bboxes ? 1 : 0) + (param_.mirror()? 1:0) +
                  ((phase_ == TRAIN && param_.crop_size())? 2 : 0);
  rn.FillRandomNumbers(count, rand_num_);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data,
                                       RandNumbers& rand_num) {
  NormalizedBBox crop_bbox;
  Transform(datum, transformed_data, &crop_bbox, rand_num);
}

template<typename Dtype>
template<typename AnnotationHandler>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob,
                                       NormalizedBBox* crop_bbox,
                                       RandNumbers& rand_num,
                                       AnnotationHandler anno_handler)
{
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob, crop_bbox, rand_num, anno_handler);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data, crop_bbox, rand_num, anno_handler);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob, RandNumbers& rand_num) {
  NormalizedBBox crop_bbox;
  Transform(datum, transformed_blob, &crop_bbox, rand_num);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const AnnotatedDatum& anno_datum,
                                       Blob<Dtype>* transformed_blob,
                                       RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
				       RandNumbers& rand_num) {
  // Transform datum.
  const Datum& datum = anno_datum.datum();
  NormalizedBBox crop_bbox;

  // We need to call TransformAnnotation after do_mirror is set, based on precalculated
  // values from RNG. RNG generates only one value for do_mirror, so the variable
  // can be set only once. Otherwise, RNG's queue will be empty.
  auto transform_annotation = [&](const bool do_resize, const bool do_mirror) -> void {
    TransformAnnotation(anno_datum, do_resize, crop_bbox, do_mirror,
                        transformed_anno_group_all);
  };

  Transform(datum, transformed_blob, &crop_bbox, rand_num, transform_annotation);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const AnnotatedDatum& anno_datum,
                                       Blob<Dtype>* transformed_blob,
                                       vector<AnnotationGroup>* transformed_anno_vec,
				       RandNumbers& rand_num) {
  RepeatedPtrField<AnnotationGroup> transformed_anno_group_all;
  Transform(anno_datum, transformed_blob, &transformed_anno_group_all, rand_num);

  for (int g = 0; g < transformed_anno_group_all.size(); ++g) {
    transformed_anno_vec->push_back(transformed_anno_group_all.Get(g));
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const AnnotatedDatum& anno_datum, 
                                       Blob<Dtype>* transformed_blob,
                                       vector<AnnotationGroup>* transformed_anno_vec) {
  Transform(anno_datum, transformed_blob, transformed_anno_vec, rand_num_);
}

template<typename Dtype>
//template<bool do_resize, bool do_mirror>
void DataTransformer<Dtype>::TransformAnnotation(const AnnotatedDatum& anno_datum,
                                                 const bool do_resize,
                                                 const NormalizedBBox& crop_bbox,
                                                 const bool do_mirror,
                                                 RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all) {
  const int img_height = anno_datum.datum().height();
  const int img_width = anno_datum.datum().width();
  if (anno_datum.type() == AnnotatedDatum_AnnotationType_BBOX) {
    // Go through each AnnotationGroup.
    for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
      const AnnotationGroup& anno_group = anno_datum.annotation_group(g);
      AnnotationGroup transformed_anno_group;
      // Go through each Annotation.
      bool has_valid_annotation = false;
      for (int a = 0; a < anno_group.annotation_size(); ++a) {
        const Annotation& anno = anno_group.annotation(a);
        const NormalizedBBox& bbox = anno.bbox();
        // Adjust bounding box annotation.
        NormalizedBBox resize_bbox = bbox;
        if (do_resize && param_.has_resize_param()) {
          CHECK_GT(img_height, 0);
          CHECK_GT(img_width, 0);
          UpdateBBoxByResizePolicy(param_.resize_param(), img_width, img_height,
                                   &resize_bbox);
        }
        if (param_.has_emit_constraint() &&
            !MeetEmitConstraint(crop_bbox, resize_bbox,
                                param_.emit_constraint())) {
          continue;
        }
        NormalizedBBox proj_bbox;
        if (ProjectBBox(crop_bbox, resize_bbox, &proj_bbox)) {
          has_valid_annotation = true;
          Annotation* transformed_anno =
              transformed_anno_group.add_annotation();
          transformed_anno->set_instance_id(anno.instance_id());
          NormalizedBBox* transformed_bbox = transformed_anno->mutable_bbox();
          transformed_bbox->CopyFrom(proj_bbox);
          if (do_mirror) {
            Dtype temp = transformed_bbox->xmin();
            transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
            transformed_bbox->set_xmax(1 - temp);
          }
          if (do_resize && param_.has_resize_param()) {
            ExtrapolateBBox(param_.resize_param(), img_height, img_width,
                crop_bbox, transformed_bbox);
          }
        }
      }
      // Save for output.
      if (has_valid_annotation) {
        transformed_anno_group.set_group_label(anno_group.group_label());
        transformed_anno_group_all->Add()->CopyFrom(transformed_anno_group);
      }
    }
  } else {
    LOG(FATAL) << "Unknown annotation type.";
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::CropImage(const Datum& datum,
                                       const NormalizedBBox& bbox,
                                       Datum* crop_datum) {
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Crop the image.
    cv::Mat crop_img;
    CropImage(cv_img, bbox, &crop_img);
    // Save the image into datum.
    EncodeCVMatToDatum(crop_img, "jpg", crop_datum);
    crop_datum->set_label(datum.label());
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Get the bbox dimension.
  NormalizedBBox clipped_bbox;
  ClipBBox(bbox, &clipped_bbox);
  NormalizedBBox scaled_bbox;
  ScaleBBox(clipped_bbox, datum_height, datum_width, &scaled_bbox);
  const int w_off = static_cast<int>(scaled_bbox.xmin());
  const int h_off = static_cast<int>(scaled_bbox.ymin());
  const int width = static_cast<int>(scaled_bbox.xmax() - scaled_bbox.xmin());
  const int height = static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());

  // Crop the image using bbox.
  crop_datum->set_channels(datum_channels);
  crop_datum->set_height(height);
  crop_datum->set_width(width);
  crop_datum->set_label(datum.label());
  crop_datum->clear_data();
  crop_datum->clear_float_data();
  crop_datum->set_encoded(false);
  const int crop_datum_size = datum_channels * height * width;
  const std::string& datum_buffer = datum.data();
  std::string buffer(crop_datum_size, ' ');
  for (int h = h_off; h < h_off + height; ++h) {
    for (int w = w_off; w < w_off + width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        int crop_datum_index = (c * height + h - h_off) * width + w - w_off;
        buffer[crop_datum_index] = datum_buffer[datum_index];
      }
    }
  }
  crop_datum->set_data(buffer);
}

template<typename Dtype>
void DataTransformer<Dtype>::CropImage(const AnnotatedDatum& anno_datum,
                                       const NormalizedBBox& bbox,
                                       AnnotatedDatum* cropped_anno_datum) {
  // Crop the datum.
  CropImage(anno_datum.datum(), bbox, cropped_anno_datum->mutable_datum());
  cropped_anno_datum->set_type(anno_datum.type());

  // Transform the annotation according to crop_bbox.
  const bool do_resize = false;
  const bool do_mirror = false;
  NormalizedBBox crop_bbox;
  ClipBBox(bbox, &crop_bbox);
  TransformAnnotation(anno_datum, do_resize, crop_bbox, do_mirror,
                      cropped_anno_datum->mutable_annotation_group());
}

template<typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const Datum& datum,
                                         const float expand_ratio,
                                         NormalizedBBox* expand_bbox,
                                         Datum* expand_datum) {
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Expand the image.
    cv::Mat expand_img;
    ExpandImage(cv_img, expand_ratio, expand_bbox, &expand_img);
    // Save the image into datum.
    EncodeCVMatToDatum(expand_img, "jpg", expand_datum);
    expand_datum->set_label(datum.label());
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Get the bbox dimension.
  int height = static_cast<int>(datum_height * expand_ratio);
  int width = static_cast<int>(datum_width * expand_ratio);
  float h_off, w_off;
  caffe_rng_uniform(1, 0.f, static_cast<float>(height - datum_height), &h_off);
  caffe_rng_uniform(1, 0.f, static_cast<float>(width - datum_width), &w_off);
  h_off = floor(h_off);
  w_off = floor(w_off);
  expand_bbox->set_xmin(-w_off/datum_width);
  expand_bbox->set_ymin(-h_off/datum_height);
  expand_bbox->set_xmax((width - w_off)/datum_width);
  expand_bbox->set_ymax((height - h_off)/datum_height);

  // Crop the image using bbox.
  expand_datum->set_channels(datum_channels);
  expand_datum->set_height(height);
  expand_datum->set_width(width);
  expand_datum->set_label(datum.label());
  expand_datum->clear_data();
  expand_datum->clear_float_data();
  expand_datum->set_encoded(false);
  const int expand_datum_size = datum_channels * height * width;
  const std::string& datum_buffer = datum.data();
  std::string buffer(expand_datum_size, ' ');
  for (int h = h_off; h < h_off + datum_height; ++h) {
    for (int w = w_off; w < w_off + datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index =
            (c * datum_height + h - h_off) * datum_width + w - w_off;
        int expand_datum_index = (c * height + h) * width + w;
        buffer[expand_datum_index] = datum_buffer[datum_index];
      }
    }
  }
  expand_datum->set_data(buffer);
}

template<typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const AnnotatedDatum& anno_datum,
                                         AnnotatedDatum* expanded_anno_datum) {
  if (!param_.has_expand_param()) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  const ExpansionParameter& expand_param = param_.expand_param();
  const float expand_prob = expand_param.prob();
  float prob;
  caffe_rng_uniform(1, 0.f, 1.f, &prob);
  if (prob > expand_prob) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  const float max_expand_ratio = expand_param.max_expand_ratio();
  if (fabs(max_expand_ratio - 1.) < 1e-2) {
    expanded_anno_datum->CopyFrom(anno_datum);
    return;
  }
  float expand_ratio;
  caffe_rng_uniform(1, 1.f, max_expand_ratio, &expand_ratio);
  // Expand the datum.
  NormalizedBBox expand_bbox;
  ExpandImage(anno_datum.datum(), expand_ratio, &expand_bbox,
              expanded_anno_datum->mutable_datum());
  expanded_anno_datum->set_type(anno_datum.type());

  // Transform the annotation according to crop_bbox.
  const bool do_resize = false;
  const bool do_mirror = false;
  TransformAnnotation(anno_datum, do_resize, expand_bbox, do_mirror,
                      expanded_anno_datum->mutable_annotation_group());
}

template<typename Dtype>
void DataTransformer<Dtype>::DistortImage(const Datum& datum,
                                          Datum* distort_datum) {
  if (!param_.has_distort_param()) {
    distort_datum->CopyFrom(datum);
    return;
  }
  // If datum is encoded, decode and crop the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
      // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Distort the image.
    cv::Mat distort_img = ApplyDistort(cv_img, param_.distort_param());
    // Save the image into datum.
    EncodeCVMatToDatum(distort_img, "jpg", distort_datum);
    distort_datum->set_label(datum.label());
    return;
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    LOG(ERROR) << "Only support encoded datum now";
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
template<typename AnnotationHandler>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob,
                                       NormalizedBBox* crop_bbox,
                                       RandNumbers& rand_num,
                                       AnnotationHandler anno_handler)
{
  const bool do_mirror = param_.mirror() && rand_num(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int transform_func_id = (do_mirror << 2) +
                          (has_mean_file << 1) +
                          has_mean_values;

  switch (transform_func_id) {
    case 0: Transform<false, false, false>(cv_img, transformed_blob, crop_bbox, rand_num);
      break;
    case 1: Transform<false, false, true >(cv_img, transformed_blob, crop_bbox, rand_num);
      break;
    case 2: Transform<false, true , false>(cv_img, transformed_blob, crop_bbox, rand_num);
      break;
    case 3: Transform<false, true , true >(cv_img, transformed_blob, crop_bbox, rand_num);
      break;
    case 4: Transform<true , false, false>(cv_img, transformed_blob, crop_bbox, rand_num);
      break;
    case 5: Transform<true , false, true >(cv_img, transformed_blob, crop_bbox, rand_num);
      break;
    case 6: Transform<true , true , false>(cv_img, transformed_blob, crop_bbox, rand_num);
      break;
    case 7: Transform<true , true , true >(cv_img, transformed_blob, crop_bbox, rand_num);
      break;
  }

  //  const bool do_resize = true;
  call_annotation_handler(anno_handler, /* do_resize*/ true, do_mirror);
}

template<typename Dtype>
template<bool do_mirror, bool has_mean_file, bool has_mean_values>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img_in,
        Blob<Dtype>* transformed_blob, NormalizedBBox* crop_bbox, RandNumbers& rand_num) {
  const cv::Mat *cv_img = &cv_img_in;
  cv::Mat resized_img;
  if (param_.has_random_resize_param()) {
#ifdef USE_OPENCV
    RandomResizeImage(cv_img_in, &resized_img);
    cv_img = &resized_img;
#else
    LOG(FATAL) << "Random image resizing requires OpenCV; compile with USE_OPENCV.";
#endif
  } else if (param_.has_random_aspect_ratio_param()) {
#ifdef USE_OPENCV
    RandomAlterAspectRatio(cv_img_in, &resized_img);
    cv_img = &resized_img;
#else
    LOG(FATAL) << "Aspect ratio changes require OpenCV; compile with USE_OPENCV.";
#endif
  }
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img->channels();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_GE(num, 1);

  CHECK(cv_img->depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();

  CHECK_GT(img_channels, 0);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
        "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
  cv::Mat cv_resized_img, cv_noised_img;
  if (param_.has_resize_param()) {
    cv_resized_img = ApplyResize(*cv_img, param_.resize_param());
  } else {
    cv_resized_img = *cv_img;
  }
  if (param_.has_noise_param()) {
    cv_noised_img = ApplyNoise(cv_resized_img, param_.noise_param());
  } else {
    cv_noised_img = cv_resized_img;
  }
  int img_height = cv_noised_img.rows;
  int img_width = cv_noised_img.cols;
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = *cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = rand_num(img_height - crop_size + 1);
      w_off = rand_num(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = (*cv_img)(roi);
  } else {
    cv_cropped_img = cv_noised_img;
  }

  // Return the normalized crop bbox.
  crop_bbox->set_xmin(Dtype(w_off) / img_width);
  crop_bbox->set_ymin(Dtype(h_off) / img_height);
  crop_bbox->set_xmax(Dtype(w_off + width) / img_width);
  crop_bbox->set_ymax(Dtype(h_off + height) / img_height);

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
         // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformInv(const Dtype* data, cv::Mat* cv_img,
                                          const int height, const int width,
                                          const int channels) {
  const Dtype scale = param_.scale();
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(channels, data_mean_.channels());
    CHECK_EQ(height, data_mean_.height());
    CHECK_EQ(width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
        "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  const int img_type = channels == 3 ? CV_8UC3 : CV_8UC1;
  cv::Mat orig_img(height, width, img_type, cv::Scalar(0, 0, 0));
  for (int h = 0; h < height; ++h) {
    uchar* ptr = orig_img.ptr<uchar>(h);
    int img_idx = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channels; ++c) {
        int idx = (c * height + h) * width + w;
        if (has_mean_file) {
          ptr[img_idx++] = static_cast<uchar>(data[idx] / scale + mean[idx]);
        } else {
          if (has_mean_values) {
            ptr[img_idx++] =
                static_cast<uchar>(data[idx] / scale + mean_values_[c]);
          } else {
            ptr[img_idx++] = static_cast<uchar>(data[idx] / scale);
          }
        }
      }
    }
  }

  if (param_.has_resize_param()) {
    *cv_img = ApplyResize(orig_img, param_.resize_param());
  } else {
    *cv_img = orig_img;
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformInv(const Blob<Dtype>* blob,
                                          vector<cv::Mat>* cv_imgs) {
  const int channels = blob->channels();
  const int height = blob->height();
  const int width = blob->width();
  const int num = blob->num();
  CHECK_GE(num, 1);
  const Dtype* image_data = blob->cpu_data();

  for (int i = 0; i < num; ++i) {
    cv::Mat cv_img;
    TransformInv(image_data, &cv_img, height, width, channels);
    cv_imgs->push_back(cv_img);
    image_data += blob->offset(1);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob,
                                       RandNumbers& rand_num) {
  NormalizedBBox crop_bbox;
  Transform(cv_img, transformed_blob, &crop_bbox, rand_num);
}

template <typename Dtype>
void DataTransformer<Dtype>::CropImage(const cv::Mat& img,
                                       const NormalizedBBox& bbox,
                                       cv::Mat* crop_img) {
  const int img_height = img.rows;
  const int img_width = img.cols;

  // Get the bbox dimension.
  NormalizedBBox clipped_bbox;
  ClipBBox(bbox, &clipped_bbox);
  NormalizedBBox scaled_bbox;
  ScaleBBox(clipped_bbox, img_height, img_width, &scaled_bbox);

  // Crop the image using bbox.
  int w_off = static_cast<int>(scaled_bbox.xmin());
  int h_off = static_cast<int>(scaled_bbox.ymin());
  int width = static_cast<int>(scaled_bbox.xmax() - scaled_bbox.xmin());
  int height = static_cast<int>(scaled_bbox.ymax() - scaled_bbox.ymin());
  cv::Rect bbox_roi(w_off, h_off, width, height);

  img(bbox_roi).copyTo(*crop_img);
}

template <typename Dtype>
void DataTransformer<Dtype>::ExpandImage(const cv::Mat& img,
                                         const float expand_ratio,
                                         NormalizedBBox* expand_bbox,
                                         cv::Mat* expand_img) {
  const int img_height = img.rows;
  const int img_width = img.cols;
  const int img_channels = img.channels();

  // Get the bbox dimension.
  int height = static_cast<int>(img_height * expand_ratio);
  int width = static_cast<int>(img_width * expand_ratio);
  float h_off, w_off;
  caffe_rng_uniform(1, 0.f, static_cast<float>(height - img_height), &h_off);
  caffe_rng_uniform(1, 0.f, static_cast<float>(width - img_width), &w_off);
  h_off = floor(h_off);
  w_off = floor(w_off);
  expand_bbox->set_xmin(-w_off/img_width);
  expand_bbox->set_ymin(-h_off/img_height);
  expand_bbox->set_xmax((width - w_off)/img_width);
  expand_bbox->set_ymax((height - h_off)/img_height);

  expand_img->create(height, width, img.type());
  expand_img->setTo(cv::Scalar(0));
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(height, data_mean_.height());
    CHECK_EQ(width, data_mean_.width());
    Dtype* mean = data_mean_.mutable_cpu_data();
    for (int h = 0; h < height; ++h) {
      uchar* ptr = expand_img->ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < img_channels; ++c) {
          int blob_index = (c * height + h) * width + w;
          ptr[img_index++] = static_cast<char>(mean[blob_index]);
        }
      }
    }
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
        "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
    vector<cv::Mat> channels(img_channels);
    cv::split(*expand_img, channels);
    CHECK_EQ(channels.size(), mean_values_.size());
    for (int c = 0; c < img_channels; ++c) {
      channels[c] = mean_values_[c];
    }
    cv::merge(channels, *expand_img);
  }

  cv::Rect bbox_roi(w_off, h_off, img_width, img_height);
  img.copyTo((*expand_img)(bbox_roi));
}

static cv::Mat ResizeImagePerShorterSize(const cv::Mat& img, int shorter_size, ResizeParameter resize_param) {
  int h = img.size().height;
  int w = img.size().width;
  resize_param.set_height(shorter_size);
  resize_param.set_width(shorter_size);
  if (h < w) {
    resize_param.set_width(int(float(w) / h * shorter_size));
  } else {
    resize_param.set_height(int(float(h) / w * shorter_size));
  }
  return ApplyResize(img, resize_param);
}

template<typename Dtype>
void DataTransformer<Dtype>::RandomResizeImage(const Datum& datum, Datum *resized_datum) {
  shared_ptr<cv::Mat> img;
  if (datum.encoded()) {
    img = shared_ptr<cv::Mat>(new cv::Mat(DecodeDatumToCVMatNative(datum)));
  } else {
    img = shared_ptr<cv::Mat>(new cv::Mat(
                                cv::Size(datum.width(), datum.height()),
                                CV_8UC(datum.channels()),
                                (void*)datum.data().data()));
  }
  cv::Mat resized_img;
  RandomResizeImage(*img, &resized_img);
  CVMatToDatum(resized_img, resized_datum);
}

template<typename Dtype>
void DataTransformer<Dtype>::RandomResizeImage(const cv::Mat& img, cv::Mat *resized_img) {
  int h = img.size().height;
  int w = img.size().width;
  int min_size = param_.random_resize_param().min_size();
  int max_size = param_.random_resize_param().max_size();
  ResizeParameter resize_param = param_.random_resize_param().resize_param();
  if (min_size == 0) min_size = std::min(h,w);
  if (max_size == 0) max_size = std::max(h,w);
  int shorter_size = rand_num_(max_size - min_size + 1) + min_size;
  *resized_img = ResizeImagePerShorterSize(img, shorter_size, resize_param);
}

template<typename Dtype>
void DataTransformer<Dtype>::RandomAlterAspectRatio(const Datum& datum, Datum *resized_datum) {
  shared_ptr<cv::Mat> img;
  if (datum.encoded()) {
    img = shared_ptr<cv::Mat>(new cv::Mat(DecodeDatumToCVMatNative(datum)));
  } else {
    img = shared_ptr<cv::Mat>(new cv::Mat(
                                cv::Size(datum.width(), datum.height()),
                                CV_8UC(datum.channels()),
                                (void*)datum.data().data()));
  }
  cv::Mat resized_img;
  RandomAlterAspectRatio(*img, &resized_img);
  CVMatToDatum(resized_img, resized_datum);
}

static float RandRatio(float min, float max, RandNumbers& rand_num) {
  return (rand_num(int((max - min) * 1000 + 1)) + min * 1000) / 1000;
}

template<typename Dtype>
void DataTransformer<Dtype>::RandomAlterAspectRatio(const cv::Mat& img, cv::Mat *resized_img) {
  const int crop_size = param_.crop_size();
  const int h = img.size().height;
  const int w = img.size().width;
  const float area = h * w;
  const float min_area_ratio = param_.random_aspect_ratio_param().min_area_ratio();
  const float max_area_ratio = param_.random_aspect_ratio_param().max_area_ratio();
  const float min_aspect_ratio_change =
    param_.random_aspect_ratio_param().aspect_ratio_change();
  CHECK(crop_size > 0);
  CHECK(max_area_ratio >= min_area_ratio);
  ResizeParameter resize_param = param_.random_aspect_ratio_param().resize_param();
  int attempt = 0;
  while (attempt++ < param_.random_aspect_ratio_param().max_attempt()) {
    float area_ratio = RandRatio(min_area_ratio, max_area_ratio, rand_num_);
    float aspect_ratio_change =
      RandRatio(min_aspect_ratio_change, 1 / min_aspect_ratio_change, rand_num_);
    float new_area = area_ratio * area;
    int new_h = int(sqrt(new_area) * aspect_ratio_change);
    int new_w = int(sqrt(new_area) / aspect_ratio_change);
    if (RandRatio(0, 1, rand_num_) < 0.5) {
      int tmp = new_h; new_h = new_w; new_w = tmp;
    }
    if (new_h <= h && new_w <= w) {
      int y = rand_num_(h - new_h + 1);
      int x = rand_num_(w - new_w + 1);
      cv::Rect roi(x, y, new_w, new_h);
      cv::Mat croppedImg = img(roi);
      resize_param.set_height(crop_size);
      resize_param.set_width(crop_size);
      *resized_img = ApplyResize(croppedImg, resize_param);
      return;
    }
  }
  *resized_img = ResizeImagePerShorterSize(img, crop_size, resize_param);
}

#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && rand_num_(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = rand_num_(input_height - crop_size + 1);
      w_off = rand_num_(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
                data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels)
        << "Specify either 1 mean_value or as many as channels: "
        << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
                           input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  int datum_height = datum.height();
  int datum_width = datum.width();

  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  if (param_.has_resize_param()) {
    InferNewSize(param_.resize_param(), datum_width, datum_height,
                 &datum_width, &datum_height);
  }

  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);

  if (param_.has_random_resize_param() || param_.has_random_aspect_ratio_param()) {
    CHECK_GT(crop_size, 0);
  } else {
    CHECK_GE(img_height, crop_size);
    CHECK_GE(img_width, crop_size);
  }

  if (param_.has_resize_param()) {
    InferNewSize(param_.resize_param(), img_width, img_height,
                 &img_width, &img_height);
  }

  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      param_.has_random_resize_param() ||
      param_.has_random_aspect_ratio_param() ||
      (phase_ == TRAIN && param_.crop_size());

  if (needs_rand) {
    rand_num_.Init();
  } else {
    rand_num_.Reset();
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::ReinitRand() {
  if (rand_num_.IsEmpty()) {
    rand_num_.Init();
  }
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
