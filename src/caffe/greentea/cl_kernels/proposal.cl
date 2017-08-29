#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

void TEMPLATE(BBoxTransformInv, Dtype)(global float *box, Dtype *delta, int anchor_shift_x,
              int anchor_shift_y, Dtype *pred_box /*output: corrected bboxes*/)
{
    float width = box[2] - box[0] + 1;
    float height = box[3] - box[1] + 1;
    float ctr_x = box[0] + 0.5f * width;
    float ctr_y = box[1] + 0.5f * height;

    float pred_ctr_x = delta[0] * width + ctr_x + anchor_shift_x;
    float pred_ctr_y = delta[1] * height + ctr_y + anchor_shift_y;
    float pred_w = exp(delta[2]) * width;
    float pred_h = exp(delta[3]) * height;

    pred_box[0] = pred_ctr_x - 0.5f * pred_w; // right
    pred_box[1] = pred_ctr_y - 0.5f * pred_h; // top
    pred_box[2] = pred_ctr_x + 0.5f * pred_w; // left
    pred_box[3] = pred_ctr_y + 0.5f * pred_h; // bottom
}

void TEMPLATE(ClipBoxes, Dtype)(Dtype *pred_box, int img_width, int img_height)
{
    //TODO: handle scale (im_info[3])
    pred_box[0] = fmax(fmin(pred_box[0], (Dtype)img_width - (Dtype)1), (Dtype)0); // right >= 0
    pred_box[1] = fmax(fmin(pred_box[1], (Dtype)img_height - (Dtype)1), (Dtype)0); // top >= 0
    pred_box[2] = fmax(fmin(pred_box[2], (Dtype)img_width - (Dtype)1), (Dtype)0); // bottom < im_shape[1]
    pred_box[3] = fmax(fmin(pred_box[3], (Dtype)img_height - (Dtype)1), (Dtype)0); // left < im_shape[0]
}

__kernel void TEMPLATE(proposalForward, Dtype)(
          global Dtype * bottom_deltas,
          global float *anchors,
          global Dtype *probs,
          int image_height,
          int image_width,
          int num_anchors,
          int feat_stride,
          int feature_map_size,
          int feature_map_width,
          int min_bbox_size,
          global Dtype *outProposal)
{
    int col_index = get_global_id(0);
    int row_index = get_global_id(1);
    int anchor_index = get_global_id(2);
    int anchor_shift_y = row_index * feat_stride;
    int anchor_shift_x = col_index * feat_stride;
    {
          int location_index = feature_map_size * anchor_index * 4 + row_index *  feature_map_width + col_index;
          global Dtype *bottom_ptr = bottom_deltas + location_index;
          Dtype bbox_delta[4];
          bbox_delta[0] = bottom_ptr[0];
          bbox_delta[1] = bottom_ptr[feature_map_size];
          bbox_delta[2] = bottom_ptr[feature_map_size * 2];
          bbox_delta[3] = bottom_ptr[feature_map_size * 3];

          int prob_index = anchor_index * feature_map_size + row_index * feature_map_width + col_index + num_anchors * feature_map_size;
          Dtype proposal_confidence = probs[prob_index];

          global float * anchor = anchors + anchor_index * 4;
          Dtype proposals[4];
          TEMPLATE(BBoxTransformInv, Dtype)(anchor, bbox_delta, anchor_shift_x, anchor_shift_y, proposals); //shift anchor and add delta fix
          TEMPLATE(ClipBoxes, Dtype)(proposals, image_width, image_height);

          Dtype bbox_w = proposals[2] - proposals[0] + 1;
          Dtype bbox_h = proposals[3] - proposals[1] + 1;
          int outI = anchor_index * feature_map_size + row_index * feature_map_width + col_index;
          if (bbox_w < min_bbox_size || bbox_h < min_bbox_size) {
            outProposal[outI * 5 + 1] = -1;
            return;
          }
          outProposal[outI * 5] = proposal_confidence;
          outProposal[outI * 5 + 1] = proposals[0];
          outProposal[outI * 5 + 2] = proposals[1];
          outProposal[outI * 5 + 3] = proposals[2];
          outProposal[outI * 5 + 4] = proposals[3];
    }
}
