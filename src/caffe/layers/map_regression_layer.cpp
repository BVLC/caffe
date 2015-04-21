//
// Created by alex on 4/21/15.
//

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

    template<typename Dtype>
    void MapRegressionLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                   const vector<Blob<Dtype>*>& top) {
        //TODO: add layer params
        beta_ = this->layer_param_.map_regression_param().beta();
    }

    template<typename Dtype>
    void MapRegressionLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top){

        vector<int> top_shape(0);  // Loss is a scalar; 0 axes.
        top[0]->Reshape(top_shape);
    }

    template <typename Dtype>
    void MapRegressionLossLayer<Dtype>::Forward_cpu(vector<Blob<Dtype> *> const &bottom,
                                                    vector<Blob<Dtype> *> const &top) {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype* label = bottom[1]->cpu_data();
        int num = bottom[0]->shape(0);
        int count = bottom[0]->count();
        int dim = bottom[0]->count(1);


    }

    template <typename Dtype>
    void MapRegressionLossLayer<Dtype>::Backward_cpu(vector<Blob<Dtype> *> const &top,
                                                     const vector<bool> &propagate_down,
                                                     vector<Blob<Dtype> *> const &bottom) {
    }

    INSTANTIATE_CLASS(MapRegressionLossLayer);
    REGISTER_LAYER_CLASS(MapRegressionLoss);
} // namespace caffe
