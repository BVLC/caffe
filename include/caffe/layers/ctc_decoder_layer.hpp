#ifndef CAFFE_CTC_DECODER_LAYER_HPP_
#define CAFFE_CTC_DECODER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief A layer that implements the decoder for a ctc
 *
 * Bottom blob is the probability of label and the sequence indicators.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class CTCDecoderLayer : public Layer<Dtype> {
 public:
  typedef vector<int> Sequence;
  typedef vector<Sequence> Sequences;

 public:
  explicit CTCDecoderLayer(const LayerParameter& param);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CTCDecoder"; }

  // probabilities (T x N x C),
  // sequence_indicators (T x N),
  // target_sequences (T X N) [optional]
  // if a target_sequence is provided, an additional accuracy top blob is
  // required
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  // sequences (terminated with negative numbers),
  // output scores [optional if 2 top blobs and bottom blobs = 2]
  // accuracy [optional, if target_sequences as bottom blob = 3]
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  const Sequences& OutputSequences() const {return output_sequences_;}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  virtual void Decode(const Blob<Dtype>* probabilities,
                      const Blob<Dtype>* sequence_indicators,
                      Sequences* output_sequences,
                      Blob<Dtype>* scores) const = 0;

  int EditDistance(const Sequence &s1, const Sequence &s2);

 protected:
  Sequences output_sequences_;
  int T_;
  int N_;
  int C_;
  int blank_index_;
  bool merge_repeated_;

  int sequence_index_;
  int score_index_;
  int accuracy_index_;
};

template <typename Dtype>
class CTCGreedyDecoderLayer : public CTCDecoderLayer<Dtype> {
 private:
  using typename CTCDecoderLayer<Dtype>::Sequences;
  using CTCDecoderLayer<Dtype>::T_;
  using CTCDecoderLayer<Dtype>::N_;
  using CTCDecoderLayer<Dtype>::C_;
  using CTCDecoderLayer<Dtype>::blank_index_;
  using CTCDecoderLayer<Dtype>::merge_repeated_;

 public:
  explicit CTCGreedyDecoderLayer(const LayerParameter& param)
      : CTCDecoderLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "CTCGreedyDecoder"; }

 protected:
  virtual void Decode(const Blob<Dtype>* probabilities,
                      const Blob<Dtype>* sequence_indicators,
                      Sequences* output_sequences,
                      Blob<Dtype>* scores) const;
};

}  // namespace caffe

#endif  // CAFFE_CTC_DECODER_LAYER_HPP_
