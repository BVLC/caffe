#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/layers/eval_detection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    class BoxData {
    public:
        int label_;
        float score_;
        vector<float> box_;
    };

    inline float sigmoid(float x)
    {
        return 1. / (1. + exp(-x));
    }

    template <typename Dtype>
    Dtype softmax_region(Dtype* input, int classes)
    {
        Dtype sum = 0;
        Dtype large = input[0];

        for (int i = 0; i < classes; ++i) {
            if (input[i] > large)
                large = input[i];
        }
        for (int i = 0; i < classes; ++i) {
            Dtype e = exp(input[i] - large);
            sum += e;
            input[i] = e;
        }
        for (int i = 0; i < classes; ++i) {
            input[i] = input[i] / sum;
        }
        return 0;
    }

    bool BoxSortDecendScore(const BoxData& box1, const BoxData& box2) {
        return box1.score_ > box2.score_;
    }

    void ApplyNms(const vector<BoxData>& boxes, vector<int>* idxes, float threshold) {
        map<int, int> idx_map;
        for (int i = 0; i < boxes.size() - 1; ++i) {
            if (idx_map.find(i) != idx_map.end()) {
                continue;
            }
            vector<float> box1 = boxes[i].box_;
            for (int j = i + 1; j < boxes.size(); ++j) {
                if (idx_map.find(j) != idx_map.end()) {
                    continue;
                }
                vector<float> box2 = boxes[j].box_;
                float iou = Calc_iou(box1, box2);
                if (iou >= threshold) {
                    idx_map[j] = 1;
                }
            }
        }
        for (int i = 0; i < boxes.size(); ++i) {
            if (idx_map.find(i) == idx_map.end()) {
                idxes->push_back(i);
            }
        }
    }

    template <typename Dtype>
    void GetGTBox(int side, vector<vector<Dtype> > boxes, map<int, vector<BoxData> >* gt_boxes) {
        //int locations = pow(side, 2);
        //for (int i = 0; i < locations; ++i) {
        for (int i = 0; i < boxes.size(); ++i) {
            //if (!label_data[locations + i]) {
            //if (label_data[i * 5 + 1] == 0) {	
          //continue;
            //	break; //maybe problem???
            //}
            vector<Dtype> box = boxes[i];
            BoxData gt_box;
            //bool difficult = (label_data[i] == 1);
            //int label = static_cast<int>(label_data[locations * 2 + i]);
            int label = box[0];
            //gt_box.difficult_ = difficult;
            gt_box.label_ = label;
            gt_box.score_ = i; //not used?
            //int box_index = locations * 3 + i * 4;
            //int box_index = i * 5 + 1;
            //LOG(INFO) << "label:" << label;
            //for (int j = 0; j < 4; ++j) {
            //  gt_box.box_.push_back(label_data[box_index + j]);
              //LOG(INFO) << "x,y,w,h:" << label_data[box_index + j];
            //}
            gt_box.box_.push_back(box[1]);
            gt_box.box_.push_back(box[2]);
            gt_box.box_.push_back(box[3]);
            gt_box.box_.push_back(box[4]);

            if (gt_boxes->find(label) == gt_boxes->end()) {
                (*gt_boxes)[label] = vector<BoxData>(1, gt_box);
            }
            else {
                (*gt_boxes)[label].push_back(gt_box);
            }
        }
    }

    template <typename Dtype>
    void GetPredBox(int side, int num_object, int num_class, Dtype* input_data, map<int, vector<BoxData> >* pred_boxes, int score_type, float nms_threshold, vector<Dtype> biases) {
        vector<BoxData> tmp_boxes;
        //int locations = pow(side, 2);
        for (int j = 0; j < side; ++j) {
            for (int i = 0; i < side; ++i) {
                for (int n = 0; n < 5; ++n)
                {
                    int index = (j * side + i) * num_object * (num_class + 1 + 4) + n * (num_class + 1 + 4);
                    float x = (i + sigmoid(input_data[index + 0])) / side;
                    float y = (j + sigmoid(input_data[index + 1])) / side;
                    float w = (exp(input_data[index + 2]) * biases[2 * n]) / side;
                    float h = (exp(input_data[index + 3]) * biases[2 * n + 1]) / side;
                    softmax_region(input_data + index + 5, num_class);

                    int pred_label = 0;
                    float max_prob = input_data[index + 5];
                    for (int c = 0; c < num_class; ++c)
                    {
                        if (max_prob < input_data[index + 5 + c])
                        {
                            max_prob = input_data[index + 5 + c];
                            pred_label = c; //0,..,19
                        }
                    }
                    BoxData pred_box;
                    pred_box.label_ = pred_label;

                    float obj_score = sigmoid(input_data[index + 4]);
                    if (score_type == 0) {
                        pred_box.score_ = obj_score;
                    }
                    else if (score_type == 1) {
                        pred_box.score_ = max_prob;
                    }
                    else {
                        pred_box.score_ = obj_score * max_prob;
                    }

                    pred_box.box_.push_back(x);
                    pred_box.box_.push_back(y);
                    pred_box.box_.push_back(w);
                    pred_box.box_.push_back(h);

                    tmp_boxes.push_back(pred_box);
                    //if (w > 1 || h > 1)
                    //	LOG(INFO)<<"Not nms pred_box:" << pred_box.label_ << " " << obj_score << " " << max_prob  << " " << pred_box.score_ << " " << pred_box.box_[0] << " " << pred_box.box_[1] << " " << pred_box.box_[2] << " " << pred_box.box_[3];	
                }
            }
        }
        /*
        for (int i = 0; i < locations; ++i) {
          int pred_label = 0;
          float max_prob = input_data[i];
          for (int j = 1; j < num_class; ++j) {
            int class_index = j * locations + i;
            if (input_data[class_index] > max_prob) {
              pred_label = j;
              max_prob = input_data[class_index];
            }
          }
          if (nms_threshold < 0) {
            if (pred_boxes->find(pred_label) == pred_boxes->end()) {
              (*pred_boxes)[pred_label] = vector<BoxData>();
            }
          }
          // LOG(INFO) << "pred_label: " << pred_label << " max_prob: " << max_prob;
          int obj_index = num_class * locations + i;
          int coord_index = (num_class + num_object) * locations + i;
          for (int k = 0; k < num_object; ++k) {
            BoxData pred_box;
            float scale = input_data[obj_index + k * locations];
            pred_box.label_ = pred_label;
            if (score_type == 0) {
              pred_box.score_ = scale;
            } else if (score_type == 1) {
              pred_box.score_ = max_prob;
            } else {
              pred_box.score_ = scale * max_prob;
            }
            int box_index = coord_index + k * 4 * locations;
            if (!constriant) {
              pred_box.box_.push_back(input_data[box_index + 0 * locations]);
              pred_box.box_.push_back(input_data[box_index + 1 * locations]);
            } else {
              pred_box.box_.push_back((i % side + input_data[box_index + 0 * locations]) / side);
              pred_box.box_.push_back((i / side + input_data[box_index + 1 * locations]) / side);
            }
            float w = input_data[box_index + 2 * locations];
            float h = input_data[box_index + 3 * locations];
            if (use_sqrt) {
              pred_box.box_.push_back(pow(w, 2));
              pred_box.box_.push_back(pow(h, 2));
            } else {
              pred_box.box_.push_back(w);
              pred_box.box_.push_back(h);
            }
            if (nms_threshold >= 0) {
              tmp_boxes.push_back(pred_box);
            } else {
              (*pred_boxes)[pred_label].push_back(pred_box);
            }
          }
        }*/
        if (nms_threshold >= 0) {
            std::sort(tmp_boxes.begin(), tmp_boxes.end(), BoxSortDecendScore);
            vector<int> idxes;
            ApplyNms(tmp_boxes, &idxes, nms_threshold);
            for (int i = 0; i < idxes.size(); ++i) {
                BoxData box_data = tmp_boxes[idxes[i]];
                //**************************************************************************************//
                if (box_data.score_ < 0.005) // from darknet
                    continue;
                //LOG(INFO)<<"box_data:" << box_data.label_ << " " << box_data.score_ << " " << box_data.box_[0] << " " << box_data.box_[1] << " " << box_data.box_[2] << " " << box_data.box_[3];
                if (pred_boxes->find(box_data.label_) == pred_boxes->end()) {
                    (*pred_boxes)[box_data.label_] = vector<BoxData>();
                }
                (*pred_boxes)[box_data.label_].push_back(box_data);
            }
        }
        else {
            for (std::map<int, vector<BoxData> >::iterator it = pred_boxes->begin(); it != pred_boxes->end(); ++it) {
                std::sort(it->second.begin(), it->second.end(), BoxSortDecendScore);
            }
        }
    }

    template <typename Dtype>
    void EvalDetectionLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        EvalDetectionParameter param = this->layer_param_.eval_detection_param();
        side_ = param.side();
        num_class_ = param.num_class();
        num_object_ = param.num_object();
        threshold_ = param.threshold();
        //sqrt_ = param.sqrt();
        //constriant_ = param.constriant();

        nms_ = param.nms();

        for (int c = 0; c < param.biases_size(); ++c) {
            biases_.push_back(param.biases(c));
        }

        switch (param.score_type()) {
        case EvalDetectionParameter_ScoreType_OBJ:
            score_type_ = 0;
            break;
        case EvalDetectionParameter_ScoreType_PROB:
            score_type_ = 1;
            break;
        case EvalDetectionParameter_ScoreType_MULTIPLY:
            score_type_ = 2;
            break;
        default:
            LOG(FATAL) << "Unknow score type.";
        }
    }

    template <typename Dtype>
    void EvalDetectionLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        int input_count = bottom[0]->count(1); //b*13*13*125
        // int label_count = bottom[1]->count(1); //b*30*5
        // outputs: classes, iou, coordinates
        int tmp_input_count = side_ * side_ * num_object_ *(num_class_ + 4 + 1); //13*13*5*25
        // label: isobj, class_label, coordinates
        //int tmp_label_count = 30 * 5;

        CHECK_EQ(input_count, tmp_input_count);
        //CHECK_EQ(label_count, tmp_label_count);

        vector<int> top_shape(2, 1);
        top_shape[0] = bottom[0]->num();
        //top_shape[1] = num_class_ + side_ * side_ * num_object_ * 4; 
        top_shape[1] = num_class_ + side_ * side_ * num_object_ * 4; //num_class_(num of each gt class) + 13 * 13 * 5 * (label + score + tp + fp)
        top[0]->Reshape(top_shape);
    }

    template <typename Dtype>
    void EvalDetectionLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        //const Dtype* input_data = bottom[0]->cpu_data();
        const Dtype* label_data = bottom[1]->cpu_data();
        //LOG(INFO) << bottom[0]->data_at(0,0,0,0) << " " << bottom[0]->data_at(0,0,0,1);  
        Blob<Dtype> swap;
        swap.Reshape(bottom[0]->num(), bottom[0]->height()*bottom[0]->width(), num_object_, bottom[0]->channels() / num_object_);

        Dtype* swap_data = swap.mutable_cpu_data();
        int index = 0;
        for (int b = 0; b < bottom[0]->num(); ++b) {
            for (int h = 0; h < bottom[0]->height(); ++h) {
                for (int w = 0; w < bottom[0]->width(); ++w) {
                    for (int c = 0; c < bottom[0]->channels(); ++c) {
                        swap_data[index++] = bottom[0]->data_at(b, c, h, w);
                    }
                }
            }
        }

        //******************************************************** Label ********************************************************//
        vector<vector<vector<Dtype> > > labels;
        labels.resize(bottom[0]->num()); //batch_size
        int num_boxes = bottom[1]->height();
        //int start = -1;

        for (int i = 0; i < num_boxes; ++i) {
            vector<Dtype> box;
            int item_id = label_data[i * 8 + 0]; //item_id
            if (item_id == -1) continue;

            Dtype xmin = label_data[i * 8 + 3];
            Dtype ymin = label_data[i * 8 + 4];
            Dtype xmax = label_data[i * 8 + 5];
            Dtype ymax = label_data[i * 8 + 6];

            Dtype cx = (xmin + xmax) / 2.0;
            Dtype cy = (ymin + ymax) / 2.0;
            Dtype w = xmax - xmin;
            Dtype h = ymax - ymin;

            box.push_back(label_data[i * 8 + 1] - 1); //0 label_id 0:bg 1-20:classes
            box.push_back(cx); //1 xmin
            box.push_back(cy); //2 ymin
            box.push_back(w); //3 xmax
            box.push_back(h); //4 ymax

            //LOG(INFO) << "item_id: "<< item_id << " label: " << box[0] << " box: [" << box[1] << " "<< box[2] << " " <<box[3] << " " << box[4] << "]";
            labels[item_id].push_back(box);
        }
        //*********************************************************Diff********************************************************//

        Dtype* top_data = top[0]->mutable_cpu_data();
        caffe_set(top[0]->count(), Dtype(0), top_data);

        for (int i = 0; i < bottom[0]->num(); ++i) {
            int input_index = i * bottom[0]->count(1);

            //int true_index = i * bottom[1]->count(1);

            int top_index = i * top[0]->count(1);
            map<int, vector<BoxData> > gt_boxes;
            GetGTBox(side_, labels[i], &gt_boxes);

            for (std::map<int, vector<BoxData > >::iterator it = gt_boxes.begin(); it != gt_boxes.end(); ++it) {
                int label = it->first;
                vector<BoxData>& g_boxes = it->second;
                for (int j = 0; j < g_boxes.size(); ++j) {
                    top_data[top_index + label] += 1; //class num++;
                }
            }

            map<int, vector<BoxData> > pred_boxes;
            //GetPredBox(side_, num_object_, num_class_, input_data + input_index, &pred_boxes, sqrt_, constriant_, score_type_, nms_);
            GetPredBox(side_, num_object_, num_class_, swap_data + input_index, &pred_boxes, score_type_, nms_, biases_);

            int index = top_index + num_class_;
            int pred_count(0);
            for (std::map<int, vector<BoxData> >::iterator it = pred_boxes.begin(); it != pred_boxes.end(); ++it) {
                int label = it->first;
                vector<BoxData>& p_boxes = it->second;
                if (gt_boxes.find(label) == gt_boxes.end()) {
                    for (int b = 0; b < p_boxes.size(); ++b) {
                        top_data[index + pred_count * 4 + 0] = p_boxes[b].label_;
                        top_data[index + pred_count * 4 + 1] = p_boxes[b].score_;
                        top_data[index + pred_count * 4 + 2] = 0; //tp
                        top_data[index + pred_count * 4 + 3] = 1; //fp
                        ++pred_count;
                    }
                    continue;
                }
                vector<BoxData>& g_boxes = gt_boxes[label];
                vector<bool> records(g_boxes.size(), false);
                for (int k = 0; k < p_boxes.size(); ++k) {
                    top_data[index + pred_count * 4 + 0] = p_boxes[k].label_;
                    top_data[index + pred_count * 4 + 1] = p_boxes[k].score_;
                    float max_iou(-1);
                    int idx(-1);
                    for (int g = 0; g < g_boxes.size(); ++g) {
                        float iou = Calc_iou(p_boxes[k].box_, g_boxes[g].box_);
                        if (iou > max_iou) {
                            max_iou = iou;
                            idx = g;
                        }
                    }
                    if (max_iou >= threshold_) {
                        if (!records[idx]) {
                            records[idx] = true;
                            top_data[index + pred_count * 4 + 2] = 1;
                            top_data[index + pred_count * 4 + 3] = 0;
                        }
                        else {
                            top_data[index + pred_count * 4 + 2] = 0;
                            top_data[index + pred_count * 4 + 3] = 1;
                        }
                    }
                    ++pred_count;
                }
            }
        }
    }

    INSTANTIATE_CLASS(EvalDetectionLayer);
    REGISTER_LAYER_CLASS(EvalDetection);

}  // namespace caffe
