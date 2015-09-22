#ifdef USE_AUDIO
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/util/fft.hpp"

namespace caffe {

template <typename Dtype>
void SpectrogramLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
    window_size_ = this->layer_param_.spectrogram_param().window_size();
    step_size_ = this->layer_param_.spectrogram_param().step_size();
}

template <typename Dtype>
void SpectrogramLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    const Dtype* bottom_labels = bottom[1]->cpu_data();
    Dtype* top_labels = top[1]->mutable_cpu_data();

    caffe_copy(bottom[1]->count(), bottom_labels, top_labels);

    int n = bottom[0]->num();
    int w = bottom[0]->width();

    int top_width = window_size_ / 2;
    int top_height = (w - window_size_ + step_size_) / step_size_;

    vector<int> top_shape(0);
    top_shape.push_back(n);
    top_shape.push_back(1);
    top_shape.push_back(top_height);
    top_shape.push_back(top_width);

    top[0]->Reshape(top_shape);

    FastFourierTransform_cpu<Dtype> fft(window_size_);

    int bottom_offset = 0;
    int top_offset = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < top_height; ++j) {
            fft.process(const_cast<Dtype*>(bottom_data + bottom_offset),
                        top_data + top_offset, window_size_);
            bottom_offset = bottom[0]->offset(i, 0, 0, j * step_size_);
            top_offset = top[0]->offset(i, 0, j, 0);
        }
    }
}

template <typename Dtype>
void SpectrogramLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
    top[1]->ReshapeLike(*bottom[1]);

    int n = bottom[0]->num();
    int w = bottom[0]->width();

    int top_width = window_size_ / 2;
    int top_height = (w - window_size_ + step_size_) / step_size_;

    vector<int> top_shape(4);
    top_shape[0] = n;
    top_shape[1] = 1;
    top_shape[2] = top_height;
    top_shape[3] = top_width;

    top[0]->Reshape(top_shape);
}

#ifdef CPU_ONLY
    STUB_GPU_FORWARD(SpectrogramLayer, Forward);
#endif

    INSTANTIATE_CLASS(SpectrogramLayer);
    REGISTER_LAYER_CLASS(Spectrogram);

}  // namespace caffe
#endif  // USE_AUDIO
