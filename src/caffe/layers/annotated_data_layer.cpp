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
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include <boost/container/vector.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {

template <typename Dtype>
AnnotatedDataLayer<Dtype>::AnnotatedDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
AnnotatedDataLayer<Dtype>::~AnnotatedDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void AnnotatedDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  label_map_file_ = anno_data_param.label_map_file();
  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }

  // Read a data point, and use it to initialize the top blob.
  AnnotatedDatum anno_datum;
  anno_datum.ParseFromString(*(reader_.full().peek()));

  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    vector<int> label_shape(4, 1);
    if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      if (anno_data_param.has_anno_type()) {
        // If anno_type is provided in AnnotatedDataParameter, replace
        // the type stored in each individual AnnotatedDatum.
        LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
        anno_type_ = anno_data_param.anno_type();
      }
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);
        label_shape[3] = 8;
      } else {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
#ifdef _OPENMP
template<typename Dtype>
void AnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  CPUTimer trans_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  AnnotatedDatum anno_datum;
  anno_datum.ParseFromString(*(reader_.full().peek()));
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  // Store transformed annotation.
  std::vector<std::vector<AnnotationGroup>> all_anno(batch_size);
  std::vector<boost::shared_ptr<AnnotatedDatum>> expand_data(batch_size, nullptr);
  std::vector<std::vector<NormalizedBBox>> sampled_bboxes(batch_size);
  boost::container::vector<bool> have_samples(batch_size, false);

  int num_bboxes = 0;

  trans_timer.Start();

// Single loop was split into two loops. SSD samples patches in the first loop, and randomly
// chooses a patch in the second loop. Sampling has to be done in the separate loop, before
// RNG precalculates random numbers in the sequential code based on number of  samples.
// TODO: correct generating random numbers in the first loop, similarly how it's done in
// the second loop and other data layers.
#pragma omp parallel if (batch_size > 1)
#pragma omp single nowait
  {
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      timer.Start();
      string* data = reader_.full().pop("Waiting for data");
      timer.Stop();
      read_time += timer.MicroSeconds();
#if !defined(_MSC_EXTENSIONS)
#pragma omp task firstprivate(item_id, data) shared(all_anno, expand_data, sampled_bboxes, have_samples)
#endif
      {
        std::unique_ptr<AnnotatedDatum> anno_datum(new AnnotatedDatum());
        anno_datum->ParseFromString(*data);
        reader_.free().push(data);
        std::unique_ptr<AnnotatedDatum> distort_datum(new AnnotatedDatum());
        boost::shared_ptr<AnnotatedDatum> expand_datum;
        if (transform_param.has_distort_param()) {
          distort_datum->CopyFrom(*anno_datum);
          this->data_transformer_->DistortImage(anno_datum->datum(),
                                                distort_datum->mutable_datum());
          if (transform_param.has_expand_param()) {
            expand_datum.reset(new AnnotatedDatum());
            this->data_transformer_->ExpandImage(*distort_datum, expand_datum.get());
          } else {
            expand_datum = std::move(distort_datum);
          }
        } else {
          if (transform_param.has_expand_param()) {
            expand_datum.reset(new AnnotatedDatum());
            this->data_transformer_->ExpandImage(*anno_datum, expand_datum.get());
          } else {
            expand_datum = std::move(anno_datum);
          }
        }
        bool has_sampled = false;
        if (batch_samplers_.size() > 0) {
          // Generate sampled bboxes from expand_datum.
          GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes[item_id]);

          if (sampled_bboxes[item_id].size() > 0) {
            has_sampled = true;
          }
        }
        expand_data[item_id] = expand_datum;
        have_samples[item_id] = has_sampled;
      }
    }
#if !defined(_MSC_EXTENSIONS)
#pragma omp taskwait
#endif
    // RNG needs to be reinitialized because in some cases, when transform params are not set
    // RNG is a NULL.
    this->data_transformer_->ReinitRand();

    for (int item_id = 0; item_id < batch_size; ++item_id) {
      PreclcRandomNumbers precalculated_rand_numbers;
      this->data_transformer_->GenerateRandNumbers(precalculated_rand_numbers, /* sample_bboxes */ have_samples[item_id]);
#if !defined(_MSC_EXTENSIONS)
#pragma omp task firstprivate(precalculated_rand_numbers, item_id) shared(num_bboxes, all_anno, expand_data, sampled_bboxes, have_samples)
#endif
      {
        boost::shared_ptr<AnnotatedDatum> sampled_datum;
        bool has_sampled = have_samples[item_id];
        if (has_sampled) {
          int rand_idx = precalculated_rand_numbers(sampled_bboxes[item_id].size());
          sampled_datum.reset(new AnnotatedDatum());
          this->data_transformer_->CropImage(*expand_data[item_id],
                                             sampled_bboxes[item_id][rand_idx],
                                             sampled_datum.get());
        } else {
          sampled_datum = expand_data[item_id];
        }
        CHECK(sampled_datum.get()  != NULL);
        Blob<Dtype> data_blob;
        data_blob.Reshape(top_shape);
        vector<int> shape =
          this->data_transformer_->InferBlobShape(sampled_datum->datum());

        if (transform_param.has_resize_param()) {
          if (transform_param.resize_param().resize_mode() ==
            ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
            DLOG(INFO) << "Has resize param";
            data_blob.Reshape(shape);
            batch->data_.Reshape(shape);
            top_data = batch->data_.mutable_cpu_data();
          } else {
            CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                             shape.begin() + 1));
          }
        } else {
          CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                shape.begin() + 1));
        }

        // Apply data transformations (mirror, scale, crop...)
        int offset = batch->data_.offset(item_id);
        data_blob.set_cpu_data(top_data + offset);
        if (this->output_labels_) {
          if (has_anno_type_) {
            // Make sure all data have same annotation type.
            CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
            if (anno_data_param.has_anno_type()) {
              sampled_datum->set_type(anno_type_);
            } else {
              CHECK_EQ(anno_type_, sampled_datum->type()) <<
                       "Different AnnotationType.";
            }
            // Transform datum and annotation_group at the same time
            this->data_transformer_->Transform(*sampled_datum,
                                               &data_blob,
                                               &all_anno[item_id],
                                               precalculated_rand_numbers);
            if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
              for (int g = 0; g < all_anno[item_id].size(); ++g) {
#pragma omp atomic
                num_bboxes += all_anno[item_id][g].annotation_size();
              }
            } else {
              LOG(FATAL) << "Unknown annotation type.";
            }
          } else {
            this->data_transformer_->Transform(sampled_datum->datum(),
                                               &data_blob,
                                               precalculated_rand_numbers);
            // Otherwise, store the label from datum.
            CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
            top_label[item_id] = sampled_datum->datum().label();
          }
        } else {
          this->data_transformer_->Transform(sampled_datum->datum(),
                                             &data_blob,
                                             precalculated_rand_numbers);
        }
      }
    }
  }

  // Store "rich" annotation if needed.
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      label_shape[3] = 8;
      if (num_bboxes == 0) {
        // Store all -1 in the label.
        label_shape[2] = 1;
        batch->label_.Reshape(label_shape);
        caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
      } else {
        // Reshape the label and store the annotation.
        label_shape[2] = num_bboxes;
        batch->label_.Reshape(label_shape);
        top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) {
            const AnnotationGroup& anno_group = anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();
              top_label[idx++] = item_id;
              top_label[idx++] = anno_group.group_label();
              top_label[idx++] = anno.instance_id();
              top_label[idx++] = bbox.xmin();
              top_label[idx++] = bbox.ymin();
              top_label[idx++] = bbox.xmax();
              top_label[idx++] = bbox.ymax();
              top_label[idx++] = bbox.difficult();
            }
          }
        }
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
  trans_timer.Stop();
  batch_timer.Stop();

  trans_time = trans_timer.MicroSeconds() - read_time;
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}
#else
template<typename Dtype>
void AnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  AnnotatedDatum anno_datum;
  anno_datum.ParseFromString(*(reader_.full().peek()));
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a anno_datum
    string* data = reader_.full().pop("Waiting for data");
    AnnotatedDatum anno_datum;
    anno_datum.ParseFromString(*data);
    reader_.free().push(data);
    read_time += timer.MicroSeconds();
    timer.Start();
    AnnotatedDatum distort_datum;
    AnnotatedDatum* expand_datum = NULL;
    if (transform_param.has_distort_param()) {
      distort_datum.CopyFrom(anno_datum);
      this->data_transformer_->DistortImage(anno_datum.datum(),
                                            distort_datum.mutable_datum());
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(distort_datum, expand_datum);
      } else {
        expand_datum = &distort_datum;
      }
    } else {
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(anno_datum, expand_datum);
      } else {
        expand_datum = &anno_datum;
      }
    }
    AnnotatedDatum* sampled_datum = NULL;
    bool has_sampled = false;
    if (batch_samplers_.size() > 0) {
      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        sampled_datum = new AnnotatedDatum();
        this->data_transformer_->CropImage(*expand_datum,
                                           sampled_bboxes[rand_idx],
                                           sampled_datum);
        has_sampled = true;
      } else {
        sampled_datum = expand_datum;
      }
    } else {
      sampled_datum = expand_datum;
    }
    CHECK(sampled_datum != NULL);
    timer.Start();
    vector<int> shape =
        this->data_transformer_->InferBlobShape(sampled_datum->datum());
    if (transform_param.has_resize_param()) {
      if (transform_param.resize_param().resize_mode() ==
          ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
        DLOG(INFO) << "Has resize param";
        this->transformed_data_.Reshape(shape);
        batch->data_.Reshape(shape);
        top_data = batch->data_.mutable_cpu_data();
      } else {
        CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
              shape.begin() + 1));
      }
    } else {
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
            shape.begin() + 1));
    }
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    vector<AnnotationGroup> transformed_anno_vec;
    if (this->output_labels_) {
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
        if (anno_data_param.has_anno_type()) {
          sampled_datum->set_type(anno_type_);
        } else {
          CHECK_EQ(anno_type_, sampled_datum->type()) <<
              "Different AnnotationType.";
        }
        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();
        this->data_transformer_->Transform(*sampled_datum,
                                           &(this->transformed_data_),
                                           &transformed_anno_vec);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
      } else {
        this->data_transformer_->Transform(sampled_datum->datum(),
                                           &(this->transformed_data_));
        // Otherwise, store the label from datum.
        CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
        top_label[item_id] = sampled_datum->datum().label();
      }
    } else {
      this->data_transformer_->Transform(sampled_datum->datum(),
                                         &(this->transformed_data_));
    }
    // clear memory
    if (has_sampled) {
      delete sampled_datum;
    }
    if (transform_param.has_expand_param()) {
      delete expand_datum;
    }
    trans_time += timer.MicroSeconds();
  }

  // Store "rich" annotation if needed.
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      label_shape[3] = 8;
      if (num_bboxes == 0) {
        // Store all -1 in the label.
        label_shape[2] = 1;
        batch->label_.Reshape(label_shape);
        caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
      } else {
        // Reshape the label and store the annotation.
        label_shape[2] = num_bboxes;
        batch->label_.Reshape(label_shape);
        top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) {
            const AnnotationGroup& anno_group = anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();
              top_label[idx++] = item_id;
              top_label[idx++] = anno_group.group_label();
              top_label[idx++] = anno.instance_id();
              top_label[idx++] = bbox.xmin();
              top_label[idx++] = bbox.ymin();
              top_label[idx++] = bbox.xmax();
              top_label[idx++] = bbox.ymax();
              top_label[idx++] = bbox.difficult();
            }
          }
        }
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}
#endif //ifdef _OPENMP

INSTANTIATE_CLASS(AnnotatedDataLayer);
REGISTER_LAYER_CLASS(AnnotatedData);

}  // namespace caffe
