#ifndef CAFFE_UTIL_SAMPLER_H_
#define CAFFE_UTIL_SAMPLER_H_

#include <vector>

#include "glog/logging.h"

#include "caffe/caffe.hpp"

namespace caffe {

// Find all annotated NormalizedBBox.
void GroupObjectBBoxes(const AnnotatedDatum& anno_datum,
                       vector<NormalizedBBox>* object_bboxes);

// Check if a sampled bbox satisfy the constraints with all object bboxes.
bool SatisfySampleConstraint(const NormalizedBBox& sampled_bbox,
                             const vector<NormalizedBBox>& object_bboxes,
                             const SampleConstraint& sample_constraint);

// Sample a NormalizedBBox given the specifictions.
void SampleBBox(const Sampler& sampler, NormalizedBBox* sampled_bbox);

// Generate samples from NormalizedBBox using the BatchSampler.
void GenerateSamples(const NormalizedBBox& source_bbox,
                     const vector<NormalizedBBox>& object_bboxes,
                     const BatchSampler& batch_sampler,
                     vector<NormalizedBBox>* sampled_bboxes);

// Generate samples from AnnotatedDatum using the BatchSampler.
// All sampled bboxes which satisfy the constraints defined in BatchSampler
// is stored in sampled_bboxes.
void GenerateBatchSamples(const AnnotatedDatum& anno_datum,
                          const vector<BatchSampler>& batch_samplers,
                          vector<NormalizedBBox>* sampled_bboxes);

}  // namespace caffe

#endif  // CAFFE_UTIL_SAMPLER_H_
