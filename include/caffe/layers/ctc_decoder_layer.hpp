#ifndef CAFFE_CTC_DECODER_LAYER_HPP_
#define CAFFE_CTC_DECODER_LAYER_HPP_

#include <algorithm>
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
  explicit CTCDecoderLayer(const LayerParameter& param)
      : Layer<Dtype>(param)
      , T_(0)
      , N_(0)
      , C_(0)
      , blank_index_(param.ctc_decoder_param().blank_index())
      , merge_repeated_(param.ctc_decoder_param().ctc_merge_repeated()) {
    }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // required additional output blob for accuracies
    if (bottom.size() == 3) {CHECK_EQ(top.size(), 2);}
  }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    Blob<Dtype>* scores = top[0];

    const Blob<Dtype>* probabilities = bottom[0];
    T_ = probabilities->shape(0);
    N_ = probabilities->shape(1);
    C_ = probabilities->shape(2);

    output_sequences_.clear();
    output_sequences_.resize(N_);

    scores->Reshape(N_, 1, 1, 1);

    if (blank_index_ < 0) {
      blank_index_ = C_ - 1;
    }

    if (top.size() == 2) {
      // Single accuracy as output
      top[1]->Reshape(1, 1, 1, 1);
    }
  }

  virtual inline const char* type() const { return "CTCDecoder"; }

  // probabilities (T x N x C),
  // sequence_indicators (T x N),
  // target_sequences (T X N) [optional]
  // if a target_sequence is provided, an additional accuracy top blob is
  // required
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

  // output scores, accuracy [optional, if target_sequences as bottom blob]
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

  const Sequences& OutputSequences() const {return output_sequences_;}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Blob<Dtype>* probabilities = bottom[0];
    const Blob<Dtype>* sequence_indicators = bottom[1];
    Blob<Dtype>* scores = top[0];

    Decode(probabilities, sequence_indicators, &output_sequences_, scores);

    if (top.size() == 2) {
      // compute accuracy
      Dtype &acc = top[1]->mutable_cpu_data()[0];
      acc = 0;

      CHECK_GE(bottom.size(), 3);  // required target sequences blob
      const Blob<Dtype>* target_sequences_data = bottom[2];
      const Dtype* ts_data = target_sequences_data->cpu_data();
      for (int n = 0; n < N_; ++n) {
        Sequence target_sequence;
        for (int t = 0; t < T_; ++t) {
          const Dtype dtarget = ts_data[target_sequences_data->offset(t, n)];
          if (dtarget < 0) {
            // sequence has finished
            break;
          }
          // round to int, just to be sure
          const int target = static_cast<int>(0.5 + dtarget);
          target_sequence.push_back(target);
        }

        const int ed = EditDistance(target_sequence, output_sequences_[n]);

        acc += ed * 1.0 /
                std::max(target_sequence.size(), output_sequences_[n].size());
      }

      acc = 1 - acc / N_;
      CHECK_GE(acc, 0);
      CHECK_LE(acc, 1);
    }
  }

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  virtual void Decode(const Blob<Dtype>* probabilities,
                      const Blob<Dtype>* sequence_indicators,
                      Sequences* output_sequences,
                      Blob<Dtype>* scores) const = 0;

  int EditDistance(const Sequence &s1, const Sequence &s2) {
    const size_t len1 = s1.size();
    const size_t len2 = s2.size();

    Sequences d(len1 + 1, Sequence(len2 + 1));

    d[0][0] = 0;
    for (size_t i = 1; i <= len1; ++i) {d[i][0] = i;}
    for (size_t i = 1; i <= len2; ++i) {d[0][i] = i;}

    for (size_t i = 1; i <= len1; ++i) {
      for (size_t j = 1; j <= len2; ++j) {
        d[i][j] = std::min(
                    std::min(
                      d[i - 1][j] + 1,
                      d[i][j - 1] + 1),
                    d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1));
      }
    }

    return d[len1][len2];
  }

 protected:
  Sequences output_sequences_;
  int T_;
  int N_;
  int C_;
  int blank_index_;
  bool merge_repeated_;
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
                      Blob<Dtype>* scores) const {
    CHECK_EQ(CHECK_NOTNULL(scores)->count(), N_);
    Dtype* score_data = scores->mutable_cpu_data();
    for (int n = 0; n < N_; ++n) {
      int prev_class_idx = -1;
      score_data[n] = 0;

      for (int t = 0; /* check at end */; ++t) {
        // get maximum probability and its index
        int max_class_idx = 0;
        const Dtype* probs = probabilities->cpu_data()
                + probabilities->offset(t, n);
        Dtype max_prob = probs[0];
        ++probs;
        for (int c = 1; c < C_; ++c, ++probs) {
            if (*probs > max_prob) {
                max_class_idx = c;
                max_prob = *probs;
            }
        }

        score_data[n] += -max_prob;

        if (max_class_idx != blank_index_
                && !(merge_repeated_&& max_class_idx == prev_class_idx)) {
            output_sequences->at(n).push_back(max_class_idx);
        }

        prev_class_idx = max_class_idx;

        if (t + 1 == T_ || sequence_indicators->data_at(t + 1, n, 0, 0) == 0) {
            // End of sequence
            break;
        }
      }
    }
  }
};

}  // namespace caffe

#endif  // CAFFE_CTC_DECODER_LAYER_HPP_
