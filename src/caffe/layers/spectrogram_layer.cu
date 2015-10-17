#ifdef USE_AUDIO
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/util/fft.hpp"

namespace caffe {

template <typename Dtype>
void SpectrogramLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    int n = bottom[0]->num();
    int c = bottom[0]->channels();
    int h = bottom[0]->height();
    int w = bottom[0]->width();

    int top_width = window_size_ / 2;
    int top_height = (w - window_size_ + step_size_) / step_size_;

    vector<int> top_shape();
    top_shape.push_back(n);
    top_shape.push_back(1);
    top_shape.push_back(top_height);
    top_shape.push_back(top_width);
    
    top[0]->Reshape(top_shape);

    FastFourierTransform_gpu<Dtype> fft(window_size_);

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

INSTANTIATE_LAYER_GPU_FORWARD(SpectrogramLayer);

}  // namespace caffe
#endif  // USE_AUDIO
