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
void YoloDetectionOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const YoloDetectionOutputParameter& yolo_detection_output_param =
      this->layer_param_.yolo_detection_output_param();
  CHECK(yolo_detection_output_param.has_num_classes()) << "Must specify num_classes";
  side_ = yolo_detection_output_param.side();
  num_classes_ = yolo_detection_output_param.num_classes();
  num_box_ = yolo_detection_output_param.num_box();
  coords_ = yolo_detection_output_param.coords();
  confidence_threshold_ = yolo_detection_output_param.confidence_threshold();
  nms_threshold_ = yolo_detection_output_param.nms_threshold();

  for (int_tp c = 0; c < yolo_detection_output_param.biases_size(); ++c) {
     biases_.push_back(yolo_detection_output_param.biases(c)); 
  } //0.73 0.87;2.42 2.65;4.30 7.04;10.24 4.59;12.68 11.87;

  if (yolo_detection_output_param.has_label_map_file())
  {
    string label_map_file = yolo_detection_output_param.label_map_file();
    if (label_map_file.empty()) 
    {
      // Ignore saving if there is no label_map_file provided.
      LOG(WARNING) << "Provide label_map_file if output results to files.";
    } 
    else 
    {
      LabelMap label_map;
      CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
          << "Failed to read label map file: " << label_map_file;
      CHECK(MapLabelToName(label_map, true, &label_to_name_))
          << "Failed to convert label to name.";
      CHECK(MapLabelToDisplayName(label_map, true, &label_to_display_name_))
          << "Failed to convert label to display name.";
    }
  }
  visualize_ = yolo_detection_output_param.visualize();
  if (visualize_) {
    visualize_threshold_ = 0.3;
    if (yolo_detection_output_param.has_visualize_threshold()) {
      visualize_threshold_ = yolo_detection_output_param.visualize_threshold();
    }
    data_transformer_.reset(
        new DataTransformer<Dtype>(this->layer_param_.transform_param(),
                                   this->phase_, this->device_));
    data_transformer_->InitRand();
    save_file_ = yolo_detection_output_param.save_file();
  }

}

template <typename Dtype>
void YoloDetectionOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  //CHECK_EQ(bottom[0]->num(), 1);
  // num() and channels() are 1.
  vector<int_tp> top_shape(2, 1);
  // Since the number of bboxes to be kept is unknown before nms, we manually
  // set it to (fake) 1.
  top_shape.push_back(1);
  // Each row is a 7 dimension vector, which stores
  // [image_id, label, confidence, x, y, w, h]
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void YoloDetectionOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int_tp num = bottom[0]->num();
  Blob<Dtype> swap;
  swap.Reshape(bottom[0]->num(), bottom[0]->height()*bottom[0]->width(), num_box_, (bottom[0]->channels() + num_box_ - 1) / num_box_);
  //std::cout<<"4"<<std::endl;  
  Dtype* swap_data = swap.mutable_cpu_data();
  int_tp index = 0;
  for (int_tp b = 0; b < bottom[0]->num(); ++b)
    for (int_tp h = 0; h < bottom[0]->height(); ++h)
      for (int_tp w = 0; w < bottom[0]->width(); ++w)
        for (int_tp c = 0; c < bottom[0]->channels(); ++c)
        {
          swap_data[index++] = bottom[0]->data_at(b,c,h,w);	
        }
    
    //CHECK_EQ(bottom[0]->data_at(0,4,1,2),swap.data_at(0,15,0,4));
    //std::cout<<"5"<<std::endl;
    //*********************************************************Activation********************************************************//
    //disp(swap);
  vector<vector< PredictionResult<Dtype> > > predicts(swap.num());
  PredictionResult<Dtype> predict;
  vector<vector<int_tp> > idxes(swap.num());
  for (int_tp b = 0; b < swap.num(); ++b){
    predicts[b].clear(); 
    idxes[b].clear();
    for (int_tp j = 0; j < side_; ++j)
      for (int_tp i = 0; i < side_; ++i)
        for (int_tp n = 0; n < num_box_; ++n){
          int_tp index = b * swap.channels() * swap.height() * swap.width() + (j * side_ + i) * swap.height() * swap.width() + n * swap.width();
          //CHECK_EQ(swap_data[index],swap.data_at(b, j * side_ + i, n, 0));
          get_region_box(swap_data, predict, biases_, n, index, i, j, side_, side_);
          predict.objScore = sigmoid(swap_data[index+4]);
          class_index_and_score(swap_data+index+5, num_classes_, predict);
          predict.confidence = predict.objScore * predict.classScore;
          if (predict.confidence >= confidence_threshold_){
            predicts[b].push_back(predict);
          }
        }
    
    if(predicts[b].size() > 0){
      ApplyNms(predicts[b], idxes[b], nms_threshold_);
    }
  }
  int_tp num_kept=0;
  for (int_tp b = 0; b < swap.num(); ++b)
	  num_kept+=idxes[b].size();
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
    }
	start_pos += idxes[b].size()*7;
  }
 if (visualize_) {
#ifdef USE_OPENCV
    vector<cv::Mat> cv_imgs;
    this->data_transformer_->TransformInv(bottom[1], &cv_imgs);
    vector<cv::Scalar> colors = GetColors(label_to_display_name_.size());
    VisualizeBBox(cv_imgs, top[0], visualize_threshold_, colors,
        label_to_display_name_, save_file_, true);
#endif
  }  
}

#ifdef CPU_ONLY
//STUB_GPU_FORWARD(YoloDetectionOutputLayer, Forward);
#endif

INSTANTIATE_CLASS(YoloDetectionOutputLayer);
REGISTER_LAYER_CLASS(YoloDetectionOutput);

}  // namespace caffe
