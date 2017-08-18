#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

//#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/layers/yolo_detection_output_layer.hpp"
//#include "caffe/layers/detection_evaluate_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
bool comp(const PredictionResult<Dtype> &a,const PredictionResult<Dtype> &b)
{
    return a.confidence>b.confidence;  //sort from big to little
}

template <typename Dtype>
void YoloDetectionOutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Blob<Dtype> swap;
  swap.Reshape(bottom[0]->num(), bottom[0]->height()*bottom[0]->width(), num_box_, (bottom[0]->channels() + num_box_ - 1) / num_box_);
  PermuteData24GPU<Dtype>(bottom[0]->count(), bottom[0]->gpu_data(),
      bottom[0]->channels(), bottom[0]->height(),bottom[0]->width(), swap.mutable_gpu_data());
  
  Dtype* swap_data = swap.mutable_cpu_data();
  vector<vector< PredictionResult<Dtype> > > predicts(swap.num());
  PredictionResult<Dtype> predict;
  int_tp num_kept=0;
  vector<vector<int_tp> > idxes(swap.num());
#ifdef _OPENMP  //liyuming mark: it only optimizes for batch>1, add -fopenmp in CMakeLists.txt CMAKE_CXX_FLAGS
   #pragma omp parallel for reduction(+:num_kept)
#endif
  for (int_tp b = 0; b < swap.num(); ++b){
    predicts[b].clear(); 
    idxes[b].clear();
    for (int_tp j = 0; j < side_; ++j)
      for (int_tp i = 0; i < side_; ++i)
        for (int_tp n = 0; n < num_box_; ++n){
          int_tp index = b * swap.channels() * swap.height() * swap.width() +
                        (j * side_ + i) * swap.height() * swap.width() +
                         n * swap.width();
          get_region_box(swap_data, predict, biases_, n, index, i, j, side_, side_);
          predict.objScore = sigmoid(swap_data[index+4]);
          class_index_and_score(swap_data+index+5, num_classes_, predict);
          predict.confidence = predict.objScore * predict.classScore;
          if (predict.confidence >= confidence_threshold_){
            predicts[b].push_back(predict);
          }
        }
    
    if(predicts[b].size() > 0){
      std::sort(predicts[b].begin(),predicts[b].end(),comp<Dtype>);
      ApplyNms(predicts[b], idxes[b], nms_threshold_);
    }
	num_kept+=idxes[b].size();
  }
  
  vector<int_tp> top_shape(2, 1);
  top_shape.push_back(num_kept);
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);

  Dtype* top_data = top[0]->mutable_cpu_data();
  int_tp start_pos=0;
  for (int_tp b = 0; b < swap.num(); ++b){
    for (int_tp i = 0; i < idxes[b].size(); i++){
      top_data[start_pos+i*7] = b;                              //Image_Id
      top_data[start_pos+i*7+1] = predicts[b][idxes[b][i]].classType; //label
      top_data[start_pos+i*7+2] = predicts[b][idxes[b][i]].confidence; //confidence
      top_data[start_pos+i*7+3] = predicts[b][idxes[b][i]].x;          
      top_data[start_pos+i*7+4] = predicts[b][idxes[b][i]].y;
      top_data[start_pos+i*7+5] = predicts[b][idxes[b][i]].w;
      top_data[start_pos+i*7+6] = predicts[b][idxes[b][i]].h;
      if(ssd_format_) {
        top_data[start_pos+i*7+1] += 1;
        top_data[start_pos+i*7+3] = predicts[b][idxes[b][i]].x - predicts[b][idxes[b][i]].w / 2.0;
        top_data[start_pos+i*7+4] = predicts[b][idxes[b][i]].y - predicts[b][idxes[b][i]].h / 2.0;
        top_data[start_pos+i*7+5] = predicts[b][idxes[b][i]].x + predicts[b][idxes[b][i]].w / 2.0;
        top_data[start_pos+i*7+6] = predicts[b][idxes[b][i]].y + predicts[b][idxes[b][i]].h / 2.0;
      }
    }
	start_pos += idxes[b].size()*7;
  }
 if (visualize_) {
#ifdef USE_OPENCV
    vector<cv::Mat> cv_imgs;
    this->data_transformer_->TransformInv(bottom[1], &cv_imgs);
    vector<cv::Scalar> colors = GetColors(label_to_display_name_.size());
    VisualizeBBox(cv_imgs, top[0], visualize_threshold_, colors,
        label_to_display_name_, save_file_, !ssd_format_);
#endif
  }  
}

INSTANTIATE_LAYER_GPU_FUNCS(YoloDetectionOutputLayer);

}  // namespace caffe
