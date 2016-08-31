#include "caffe/layers/ctc_decoder_layer.hpp"

namespace caffe {

INSTANTIATE_CLASS(CTCDecoderLayer);

INSTANTIATE_CLASS(CTCGreedyDecoderLayer);
REGISTER_LAYER_CLASS(CTCGreedyDecoder);

}  // namespace caffe
