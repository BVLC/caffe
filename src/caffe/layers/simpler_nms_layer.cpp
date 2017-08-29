#include <algorithm>
#include <math.h>
#include <tuple>
#include "caffe/gen_anchors.hpp"
#include "caffe/layers/fast_rcnn_layers.hpp"


namespace caffe
{
template <typename Dtype>
void SimplerNMSLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    const SimplerNMSParameter& nms_param = this->layer_param_.simpler_nms_param();
    //if (this->layer_param_.has_simpler_nms_param()) //TODO: Check why the function returns false
    //{
    max_proposals_ = nms_param.max_num_proposals();
    prob_threshold_ = nms_param.cls_threshold();
    iou_threshold_ = nms_param.iou_threshold();
    min_bbox_size_ = nms_param.min_bbox_size();
    //TODO: handle feat_stride
    CHECK(nms_param.feat_stride() == 16) << this->type() << " layer currently doesn't support other feat_stride value than 16.";
    feat_stride_ = nms_param.feat_stride();
    pre_nms_topN_ = nms_param.pre_nms_topn();
    post_nms_topN_ = nms_param.post_nms_topn();

    vector<float> scales(nms_param.scale_size());

    for (int i = 0 ; i < nms_param.scale_size() ; i++) {
        scales[i] = nms_param.scale(i);
    }

    vector<float> default_ratios(3);
    default_ratios[0] = 0.5f;
    default_ratios[1] = 1.0f;
    default_ratios[2] = 2.0f;

    unsigned int default_size = 16;

    anchors_blob_.Reshape(default_ratios.size(), scales.size(), sizeof(anchor) / sizeof(float), 1);
    anchor *anchors = (anchor*) anchors_blob_.mutable_cpu_data();
    GenerateAnchors(default_size, default_ratios, scales, anchors);
}

template <typename Dtype>
void SimplerNMSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    int anchors_num = anchors_blob_.shape(0) * anchors_blob_.shape(1);
    const anchor* anchors = (anchor*) anchors_blob_.cpu_data();

    // feat map sizes
    int fm_w = bottom[0]->shape(3);
    int fm_h = bottom[0]->shape(2);
    int fm_sz = fm_w * fm_h;

    // original input image to the graph (after possible scaling etc.) so that coordinates are valid for it
    int img_w = (int)bottom[2]->cpu_data()[1];
    int img_h = (int)bottom[2]->cpu_data()[0];

    //TODO(ruv): what is it being multipied by, here??
    int scaled_min_bbox_size = min_bbox_size_ * (int)bottom[2]->cpu_data()[2];

    const Dtype* bottom_cls_scores = bottom[0]->cpu_data();
    const Dtype* bottom_delta_pred = bottom[1]->cpu_data();
    Dtype * top_data = top[0]->mutable_cpu_data();

    std::vector<simpler_nms_proposal_t> sorted_proposals_confidence;
    for (unsigned y = 0; y < fm_h; ++y)
    {
        int anchor_shift_y = y * feat_stride_;

        for (unsigned x = 0; x < fm_w; ++x)
        {
            int anchor_shift_x = x * feat_stride_;
            int location_index = y * fm_w + x;

            // we assume proposals are grouped by window location
            for (int anchor_index = 0; anchor_index < anchors_num ; anchor_index++)
            {
                Dtype dx0 = bottom_delta_pred[location_index + fm_sz * (anchor_index * 4 + 0)];
                Dtype dy0 = bottom_delta_pred[location_index + fm_sz * (anchor_index * 4 + 1)];
                Dtype dx1 = bottom_delta_pred[location_index + fm_sz * (anchor_index * 4 + 2)];
                Dtype dy1 = bottom_delta_pred[location_index + fm_sz * (anchor_index * 4 + 3)];
                simpler_nms_delta_t bbox_delta { dx0, dy0, dx1, dy1 };

                Dtype proposal_confidence =
                    bottom_cls_scores[location_index + fm_sz * (anchor_index + anchors_num * 1)];

                simpler_nms_roi_t tmp_roi = simpler_nms_gen_bbox(anchors[anchor_index], bbox_delta, anchor_shift_x, anchor_shift_y);
                simpler_nms_roi_t roi = tmp_roi.clamp({ 0, 0, Dtype(img_w - 1), Dtype(img_h - 1) });

                int bbox_w = roi.x1 - roi.x0 + 1;
                int bbox_h = roi.y1 - roi.y0 + 1;

                if (bbox_w >= scaled_min_bbox_size && bbox_h >= scaled_min_bbox_size)
                {
                    simpler_nms_proposal_t proposal { roi, proposal_confidence, sorted_proposals_confidence.size() };
                    sorted_proposals_confidence.push_back(proposal);
                }
            }
        }
    }

    sort_and_keep_at_most_top_n(sorted_proposals_confidence, pre_nms_topN_);
    auto res = simpler_nms_perform_nms(sorted_proposals_confidence, iou_threshold_, post_nms_topN_);

    size_t res_num_rois = res.size();
    for (size_t i = 0; i < res_num_rois; ++i)
    {
        top_data[5 * i + 0] = 0;    // roi_batch_ind, always zero on test time
        top_data[5 * i + 1] = res[i].x0;
        top_data[5 * i + 2] = res[i].y0;
        top_data[5 * i + 3] = res[i].x1;
        top_data[5 * i + 4] = res[i].y1;
    }

    top[0]->Reshape(vector<int>{ (int)res_num_rois, 5 });
}

template <typename Dtype>
std::vector< typename SimplerNMSLayer<Dtype>::simpler_nms_roi_t >
SimplerNMSLayer<Dtype>::simpler_nms_perform_nms(
        const std::vector<simpler_nms_proposal_t>& proposals,
        float iou_threshold,
        size_t top_n)
{
//TODO(ruv): can I mark the 1st arg, proposals as const? ifndef DONT_PRECALC_AREA, i can
//TODO(ruv): is it better to do the precalc or not? since we need to fetch the floats from memory anyway for -
//           intersect calc, it's only a question of whether it's faster to do (f-f)*(f-f) or fetch another val
#define DONT_PRECALC_AREA

#ifndef DONT_PRECALC_AREA
    std::vector<Dtype> areas;
    areas.reserve(proposals.size());
    std::transform(proposals.begin(), proposals.end(), areas.begin(), [](const simpler_nms_proposals_t>& v)
    {
        return v.roi.area();
    });
#endif

    std::vector<simpler_nms_roi_t> res;
    res.reserve(top_n);
#ifdef DONT_PRECALC_AREA
    for (const auto & prop : proposals)
    {
        const auto bbox = prop.roi;
        const Dtype area = bbox.area();
#else
    size_t proposal_count = proposals.size();
    for (size_t proposalIndex = 0; proposalIndex < proposal_count; ++proposalIndex)
    {
        const auto & bbox = proposals[proposalIndex].roi;
#endif

        // For any realistic WL, this condition is true for all top_n values anyway
        if (prop.confidence > 0)
        {
            bool overlaps = std::any_of(res.begin(), res.end(), [&](const simpler_nms_roi_t& res_bbox)
            {
                Dtype interArea = bbox.intersect(res_bbox).area();
#ifdef DONT_PRECALC_AREA
                Dtype unionArea = res_bbox.area() + area - interArea;
#else
                Dtype unionArea = res_bbox.area() + areas[proposalIndex] - interArea;
#endif

                return interArea > iou_threshold * unionArea;
            });

            if (! overlaps)
            {
                res.push_back(bbox);
                if (res.size() == top_n) break;
            }
        }
    }

    return res;
}

template <typename Dtype>
inline void SimplerNMSLayer<Dtype>::sort_and_keep_at_most_top_n(
        std::vector<simpler_nms_proposal_t>& proposals,
        size_t top_n)
{
    const auto cmp_fn = [](const simpler_nms_proposal_t& a,
                           const simpler_nms_proposal_t& b)
    {
        return a.confidence > b.confidence || (a.confidence == b.confidence && a.ord > b.ord);
    };

    if (proposals.size() > top_n)
    {
        std::partial_sort(proposals.begin(), proposals.begin() + top_n, proposals.end(), cmp_fn);
        proposals.resize(top_n);
    }
    else
        std::sort(proposals.begin(), proposals.end(), cmp_fn);
}

template <typename Dtype>
inline typename SimplerNMSLayer<Dtype>::simpler_nms_roi_t
SimplerNMSLayer<Dtype>::simpler_nms_gen_bbox(
        const anchor& box,
        const simpler_nms_delta_t& delta,
        int anchor_shift_x,
        int anchor_shift_y)
{
    auto anchor_w = box.end_x - box.start_x + 1;
    auto anchor_h = box.end_y - box.start_y + 1;
    auto center_x = box.start_x + anchor_w * .5f;
    auto center_y = box.start_y + anchor_h *.5f;

    Dtype pred_center_x = delta.shift_x * anchor_w + center_x + anchor_shift_x;
    Dtype pred_center_y = delta.shift_y * anchor_h + center_y + anchor_shift_y;
    Dtype half_pred_w = std::exp(delta.log_w) * anchor_w * .5f;
    Dtype half_pred_h = std::exp(delta.log_h) * anchor_h * .5f;

    return { pred_center_x - half_pred_w,
             pred_center_y - half_pred_h,
             pred_center_x + half_pred_w,
             pred_center_y + half_pred_h };
}

template <typename Dtype>
void SimplerNMSLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    //NOT_IMPLEMENTED;
};

#ifdef CPU_ONLY
    STUB_GPU(SimplerNMSLayer);
#endif

INSTANTIATE_CLASS(SimplerNMSLayer);
REGISTER_LAYER_CLASS(SimplerNMS);

}
