#include <algorithm>
#include <vector>

#include "caffe/util/bbox_util.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {

void GroupObjectBBoxes(const AnnotatedDatum& anno_datum,
                       vector<NormalizedBBox>* object_bboxes) {
  object_bboxes->clear();
  for (int i = 0; i < anno_datum.annotation_group_size(); ++i) {
    const AnnotationGroup& anno_group = anno_datum.annotation_group(i);
    for (int j = 0; j < anno_group.annotation_size(); ++j) {
      const Annotation& anno = anno_group.annotation(j);
      object_bboxes->push_back(anno.bbox());
    }
  }
}

bool SatisfySampleConstraint(const NormalizedBBox& sampled_bbox,
                             const vector<NormalizedBBox>& object_bboxes,
                             const SampleConstraint& sample_constraint) {
  bool has_jaccard_overlap = sample_constraint.has_min_jaccard_overlap() ||
      sample_constraint.has_max_jaccard_overlap();
  bool has_sample_coverage = sample_constraint.has_min_sample_coverage() ||
      sample_constraint.has_max_sample_coverage();
  bool has_object_coverage = sample_constraint.has_min_object_coverage() ||
      sample_constraint.has_max_object_coverage();
  bool satisfy = !has_jaccard_overlap && !has_sample_coverage &&
      !has_object_coverage;
  if (satisfy) {
    // By default, the sampled_bbox is "positive" if no constraints are defined.
    return true;
  }
  // Check constraints.
  bool found = false;
  for (int i = 0; i < object_bboxes.size(); ++i) {
    const NormalizedBBox& object_bbox = object_bboxes[i];
    // Test jaccard overlap.
    if (has_jaccard_overlap) {
      const float jaccard_overlap = JaccardOverlap(sampled_bbox, object_bbox);
      if (sample_constraint.has_min_jaccard_overlap() &&
          jaccard_overlap < sample_constraint.min_jaccard_overlap()) {
        continue;
      }
      if (sample_constraint.has_max_jaccard_overlap() &&
          jaccard_overlap > sample_constraint.max_jaccard_overlap()) {
        continue;
      }
      found = true;
    }
    // Test sample coverage.
    if (has_sample_coverage) {
      const float sample_coverage = BBoxCoverage(sampled_bbox, object_bbox);
      if (sample_constraint.has_min_sample_coverage() &&
          sample_coverage < sample_constraint.min_sample_coverage()) {
        continue;
      }
      if (sample_constraint.has_max_sample_coverage() &&
          sample_coverage > sample_constraint.max_sample_coverage()) {
        continue;
      }
      found = true;
    }
    // Test object coverage.
    if (has_object_coverage) {
      const float object_coverage = BBoxCoverage(object_bbox, sampled_bbox);
      if (sample_constraint.has_min_object_coverage() &&
          object_coverage < sample_constraint.min_object_coverage()) {
        continue;
      }
      if (sample_constraint.has_max_object_coverage() &&
          object_coverage > sample_constraint.max_object_coverage()) {
        continue;
      }
      found = true;
    }
    if (found) {
      return true;
    }
  }
  return found;
}

void SampleBBox(const Sampler& sampler, NormalizedBBox* sampled_bbox) {
  // Get random scale.
  CHECK_GE(sampler.max_scale(), sampler.min_scale());
  CHECK_GT(sampler.min_scale(), 0.);
  CHECK_LE(sampler.max_scale(), 1.);
  float scale;
  caffe_rng_uniform(1, sampler.min_scale(), sampler.max_scale(), &scale);

  // Get random aspect ratio.
  CHECK_GE(sampler.max_aspect_ratio(), sampler.min_aspect_ratio());
  CHECK_GT(sampler.min_aspect_ratio(), 0.);
  CHECK_LT(sampler.max_aspect_ratio(), FLT_MAX);
  float aspect_ratio;
  caffe_rng_uniform(1, sampler.min_aspect_ratio(), sampler.max_aspect_ratio(),
      &aspect_ratio);

  aspect_ratio = std::max<float>(aspect_ratio, std::pow(scale, 2.));
  aspect_ratio = std::min<float>(aspect_ratio, 1 / std::pow(scale, 2.));

  // Figure out bbox dimension.
  float bbox_width = scale * sqrt(aspect_ratio);
  float bbox_height = scale / sqrt(aspect_ratio);

  // Figure out top left coordinates.
  float w_off, h_off;
  caffe_rng_uniform(1, 0.f, 1 - bbox_width, &w_off);
  caffe_rng_uniform(1, 0.f, 1 - bbox_height, &h_off);

  sampled_bbox->set_xmin(w_off);
  sampled_bbox->set_ymin(h_off);
  sampled_bbox->set_xmax(w_off + bbox_width);
  sampled_bbox->set_ymax(h_off + bbox_height);
}

void GenerateSamples(const NormalizedBBox& source_bbox,
                     const vector<NormalizedBBox>& object_bboxes,
                     const BatchSampler& batch_sampler,
                     vector<NormalizedBBox>* sampled_bboxes) {
  int found = 0;
  for (int i = 0; i < batch_sampler.max_trials(); ++i) {
    if (batch_sampler.has_max_sample() &&
        found >= batch_sampler.max_sample()) {
      break;
    }
    // Generate sampled_bbox in the normalized space [0, 1].
    NormalizedBBox sampled_bbox;
    SampleBBox(batch_sampler.sampler(), &sampled_bbox);
    // Transform the sampled_bbox w.r.t. source_bbox.
    LocateBBox(source_bbox, sampled_bbox, &sampled_bbox);
    // Determine if the sampled bbox is positive or negative by the constraint.
    if (SatisfySampleConstraint(sampled_bbox, object_bboxes,
                                batch_sampler.sample_constraint())) {
      ++found;
      sampled_bboxes->push_back(sampled_bbox);
    }
  }
}

void GenerateBatchSamples(const AnnotatedDatum& anno_datum,
                          const vector<BatchSampler>& batch_samplers,
                          vector<NormalizedBBox>* sampled_bboxes) {
  sampled_bboxes->clear();
  vector<NormalizedBBox> object_bboxes;
  GroupObjectBBoxes(anno_datum, &object_bboxes);
  for (int i = 0; i < batch_samplers.size(); ++i) {
    if (batch_samplers[i].use_original_image()) {
      NormalizedBBox unit_bbox;
      unit_bbox.set_xmin(0);
      unit_bbox.set_ymin(0);
      unit_bbox.set_xmax(1);
      unit_bbox.set_ymax(1);
      GenerateSamples(unit_bbox, object_bboxes, batch_samplers[i],
                      sampled_bboxes);
    }
  }
}

}  // namespace caffe
