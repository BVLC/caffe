#include "caffe/layers/ctc_loss_layer.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace caffe {

/**
 * @brief Implodes a vector into a string
 * @param in The input vector
 * @param delim The delimiter
 * @return string out of the input vector
 */
std::string imploded(const std::vector<int> &in,
                     const std::string &delim = " ") {
  std::stringstream out;
  out << in[0];
  for (int i = 1; i < in.size(); ++i) {
    out << delim << in[i];
  }
  return out.str();
}

/**
 * @brief Converts a Dtype memory label sequence into a int vector
 * @param label Pointer to the start of the sequence
 * @param label_size The size of the label (number of elements to read from
 *                   label)
 * @param label_incr The offset to the next label in the sequence in the raw
 *                   data input of label
 * @return int vector containing the sequence
 */
template <typename Dtype>
vector<int> extract_label_sequence(const Dtype* label,
                                   int label_size,
                                   int label_incr) {
  vector<int> out(label_size);
  for (int i = 0; i < label_size; ++i) {
    out[i] = static_cast<int>(*label + 0.5);
    label += label_incr;
  }
  return out;
}

// Probability calculation utils.
// Note that only in double space.
// The c++ standard before c++11 does not support templates on variables yet.
// When setting c++11 to required standard add template <Dtype> and replace
// double.

/// Zero probability in log space
static const double kLogZero = -std::numeric_limits<double>::infinity();

/**
 * @brief Adds two log probabilities. This equates a multiplication of
 *        probabilities in normal space.
 * @param log_prob_1 The first probability
 * @param log_prob_2 The second probability
 * @returns The added log prob
 */
inline double LogSumExp(double log_prob_1, double log_prob_2) {
  // Always have 'b' be the smaller number to avoid the exponential from
  // blowing up.
  if (log_prob_1 == kLogZero && log_prob_2 == kLogZero) {
    return kLogZero;
  } else {
    return (log_prob_1 > log_prob_2)
            ? log_prob_1 + log1p(exp(log_prob_2 - log_prob_1))
            : log_prob_2 + log1p(exp(log_prob_1 - log_prob_2));
  }
}


template <typename Dtype>
CTCLossLayer<Dtype>::CTCLossLayer(const LayerParameter& param)
     : LossLayer<Dtype>(param),
       T_(0),
       N_(0),
       C_(0) {
  output_delay_ = param.ctc_loss_param().output_delay();
  blank_index_  = param.ctc_loss_param().blank_index();
  preprocess_collapse_repeated_ =
          param.ctc_loss_param().preprocess_collapse_repeated();
  ctc_merge_repeated_ = param.ctc_loss_param().ctc_merge_repeated();
  loss_calculation_t_ = param.ctc_loss_param().loss_calculation_t();
}

template <typename Dtype>
CTCLossLayer<Dtype>::~CTCLossLayer() {
  // clear all data blobs
  CHECK_EQ(N_, log_alpha_.size());
  CHECK_EQ(N_, log_beta_.size());
  CHECK_EQ(N_, l_primes_.size());
  CHECK_EQ(N_, y_.size());
  for (int n = 0; n < N_; ++n) {
    // dummy shapes
    delete log_alpha_[n];
    delete log_beta_[n];
    delete l_primes_[n];
    delete y_[n];
  }
  log_alpha_.clear();
  log_beta_.clear();
  l_primes_.clear();
  y_.clear();
}

template <typename Dtype>
void CTCLossLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  loss_calculation_t_ = 0;

  const Blob<Dtype>* probs = bottom[0];
  const Blob<Dtype>* seq_ind = bottom[1];
  const Blob<Dtype>* label_seq = bottom[2];


  T_ = probs->num();
  N_ = probs->channels();
  C_ = probs->height();
  CHECK_EQ(probs->width(), 1);

  CHECK_EQ(T_, seq_ind->num());
  CHECK_EQ(N_, seq_ind->channels());
  CHECK_EQ(N_, label_seq->channels());

  if (blank_index_ < 0) {
    // select the last label as default label if user did not specify
    // one with the blank_index parameter.
    blank_index_ = C_ - 1;
  }

  // resize data storage blobs for each sequence
  seq_len_.Reshape(N_, 1, 1, 1);
  label_len_.Reshape(N_, 1, 1, 1);
  log_p_z_x_.Reshape(N_, 1, 1, 1);


  // resize alpha and beta containers to the required input sequences length
  log_alpha_.resize(N_);
  log_beta_.resize(N_);
  l_primes_.resize(N_);
  y_.resize(N_);

  for (int n = 0; n < N_; ++n) {
    // dummy shapes
    log_alpha_[n] = new CTCVariables(1, 1, 1, 1);
    log_beta_[n] = new CTCVariables(1, 1, 1, 1);
    l_primes_[n] = new SequenceBlob(1, 1, 1, 1);
    y_[n] = new Blob<Dtype>(1, 1, 1, 1);
  }
}

template <typename Dtype>
void CTCLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void CTCLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  Blob<Dtype> *data_blob = bottom[0];
  const Blob<Dtype> *seq_ind_blob = bottom[1];
  const Blob<Dtype> *target_seq_blob = bottom[2];

  CHECK_EQ(data_blob->num(), seq_ind_blob->num());
  CHECK_EQ(data_blob->num(), target_seq_blob->num());

  CHECK_EQ(data_blob->channels(), seq_ind_blob->channels());
  CHECK_EQ(data_blob->channels(), target_seq_blob->channels());

  // compute the sequence length and label length
  int* seq_len = seq_len_.mutable_cpu_data();
  int* label_len = label_len_.mutable_cpu_data();
  for (int n = 0; n < N_; ++n) {
    seq_len[n] = T_;  // default value is maximal allowed length
    label_len[n] = T_;  // default value is maximal allowed length

    const Dtype *seq = seq_ind_blob->cpu_data() + n;
    const Dtype *label = target_seq_blob->cpu_data() + n;

    // sequence indicators start with seq == 0.0 to indicate the start of a
    // sequence. Skip at t = 0, so start at t = 1
    seq += seq_ind_blob->channels();
    for (int t = 1; t < T_; ++t) {
      if (static_cast<int>(*seq + 0.5) == 0) {
        seq_len[n] = t;
        break;
      }
      seq += seq_ind_blob->channels();
    }

    // label indicators are negative if the sequence has ended
    for (int t = 0; t < T_; ++t) {
      if (*label < 0.0) {
        label_len[n] = t;
        break;
      }
      label += target_seq_blob->channels();
    }

    CHECK_LE(label_len[n], seq_len[n])
         << "The label length must be smaller or equals the sequence length!";
  }


  // compute loss (in forward pass), and store computed results for backward
  // pass
  Dtype &loss = top[0]->mutable_cpu_data()[0];

  CalculateLoss(&seq_len_,
                target_seq_blob,
                &label_len_,
                data_blob,
                &log_p_z_x_,
                &l_primes_,
                &y_,
                preprocess_collapse_repeated_,
                ctc_merge_repeated_,
                &loss,
                true);

  // normalize by number of parallel batches
  loss /= N_;
}

template <typename Dtype>
void CTCLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {
  CHECK_EQ(propagate_down[0], true)
        << "Required to propagate to probabilities";
  CHECK_EQ(propagate_down[1], false)
        << "Cannot propagate to sequence indicators";
  CHECK_EQ(propagate_down[2], false)
        << "Cannot propagate to target label sequence";

  Blob<Dtype> *data_blob = bottom[0];

  const int* seq_len = seq_len_.cpu_data();
  const double* log_p_z_x = log_p_z_x_.cpu_data();

  // clear all diffs in data blob
  caffe_set<Dtype>(data_blob->count(), 0, data_blob->mutable_cpu_diff());

  // for each batch compute the gradient using the alpha and beta variables,
  // y_, p_z_x and l_primes  that were computed in the forward pass
  for (int b = 0; b < N_; ++b) {
    // We compute the derivative if needed
    CalculateGradient(b,
                      seq_len[b],
                      l_primes_[b],
                      y_[b],
                      log_alpha_[b],
                      log_beta_[b],
                      log_p_z_x[b],
                      data_blob);
  }
}

template <typename Dtype>
void CTCLossLayer<Dtype>::CalculateLoss(
        const LengthBlob *seq_len_blob,
        const Blob<Dtype>* target_seq_blob,
        const LengthBlob *target_seq_len_blob,
        Blob<Dtype>* data_blob,
        ProbBlob *log_p_z_x,
        LabelSequences *l_primes,
        vector<Blob<Dtype> *> *y,
        bool preprocess_collapse_repeated,
        bool ctc_merge_repeated,
        Dtype* loss,
        bool requires_backprob) const {
  CHECK(seq_len_blob);
  CHECK(target_seq_blob);
  CHECK(target_seq_len_blob);
  CHECK(data_blob);
  CHECK(loss);

  const int *seq_len = seq_len_blob->cpu_data();

  const int num_time_steps = T_;
  const int batch_size = N_;
  const int num_classes = C_;

  // check validity of data
  CHECK_EQ(data_blob->num(), num_time_steps);
  CHECK_EQ(data_blob->channels(), batch_size);
  CHECK_EQ(data_blob->height(), num_classes);
  CHECK_EQ(data_blob->width(), 1);

  // check validity of sequence_length arrays
  int max_seq_len = seq_len[0];
  for (int b = 0; b < batch_size; ++b) {
    CHECK_GE(seq_len[b], 0);
    CHECK_LE(seq_len[b], num_time_steps);
    max_seq_len = std::max(max_seq_len, seq_len[b]);
  }

  // set loss to 0
  *loss = 0;

  // calculate the modified label sequence l' for each batch element,
  // and calculate the maximum necessary allocation size.
  int max_u_prime = 0;
  PopulateLPrimes(preprocess_collapse_repeated,
                  batch_size,
                  num_classes,
                  *seq_len_blob,
                  *target_seq_blob,
                  *target_seq_len_blob,
                  &max_u_prime,
                  l_primes);

  // Compute loss and gradients
  // ============================================================

  // TODO: this can be parallelized
  for (int b = 0; b < batch_size; ++b) {
    const int seq_len_b = seq_len[b];
    if (seq_len_b == 0) {
      continue;  // zero length, no gradients or loss to be computed
    }

    const SequenceBlob *l_prime = l_primes->at(b);

    const int b_T = seq_len_b - output_delay_;

    // alpha and beta reshape and access
    CTCVariables* log_alpha_b = log_alpha_[b];
    CTCVariables* log_beta_b = log_beta_[b];
    log_alpha_b->Reshape(l_prime->count(), b_T, 1, 1);
    log_beta_b->Reshape(l_prime->count(), b_T, 1, 1);
    double *log_alpha_data = log_alpha_b->mutable_cpu_data();
    double *log_beta_data = log_beta_b->mutable_cpu_data();

    // Work matrices, pre-allocated to the size required by this batch item
    const Dtype* data_start = data_blob->cpu_data();
    Blob<Dtype>* y_b = y->at(b);
    y_b->Reshape(seq_len_b, C_, 1, 1);
    Dtype* y_start = y_b->mutable_cpu_data();

    // compute softmax (until sequence length is sufficient)
    for (int t = 0; t < seq_len_b; ++t) {
      const Dtype* data = data_start + data_blob->offset(t, b);
      Dtype* y_out_start = y_start + y_b->offset(t);
      Dtype max_coeff = *data;
      // get max coeff
      for (const Dtype* c_data = data + 1; c_data != data + C_; ++c_data) {
          max_coeff = std::max(max_coeff, *c_data);
      }
      // calc exp and its sum
      Dtype sum = 0;
      Dtype* y_out = y_out_start;
      for (const Dtype* c_data = data; c_data != data + C_; ++c_data) {
        *y_out = exp(*c_data - max_coeff);
        sum += *y_out++;
      }
      // division by sum
      for (y_out = y_out_start; y_out != y_out_start + C_; ++y_out) {
        *y_out /= sum;
      }
    }

    // Compute forward, backward variables.
    CalculateForwardVariables(l_prime, y_b, ctc_merge_repeated, log_alpha_b);
    CalculateBackwardVariables(l_prime, y_b, ctc_merge_repeated, log_beta_b);

    // the lost is computed as the log(p(z|x)) between the target and the
    // prediction. Do lazy evaluation of log_prob here.
    double& log_p_z_x_b = log_p_z_x->mutable_cpu_data()[b];
    log_p_z_x_b = kLogZero;
    const int loss_calc_t = std::max(0, std::min(loss_calculation_t_, b_T - 1));
    for (int u = 0; u < l_prime->count(); ++u) {
        int offset = log_alpha_b->offset(u, loss_calc_t);
        log_p_z_x_b = LogSumExp(log_p_z_x_b,
                               log_alpha_data[offset] + log_beta_data[offset]);
    }

    // use negative loss for display
    *loss += -log_p_z_x_b;
  }
}

template <typename Dtype>
void CTCLossLayer<Dtype>::PopulateLPrimes(bool preprocess_collapse_repeated,
                                          int N,
                                          int num_classes,
                                          const LengthBlob &seq_len,
                                          const Blob<Dtype>& labels,
                                          const LengthBlob &label_len,
                                          int *max_u_prime,
                                          LabelSequences* l_primes) const {
  CHECK_EQ(seq_len.num(), N);
  CHECK_EQ(seq_len.count(), seq_len.num());  // shape must be N x 1 x 1 x 1
  CHECK_EQ(labels.channels(), N);
  CHECK_EQ(label_len.num(), N);
  CHECK_EQ(label_len.count(), label_len.num());  // shape must be N x 1 x 1 x 1
  CHECK(max_u_prime);
  CHECK(l_primes);

  *max_u_prime = 0;  // keep track of longest l' modified label sequence.

  const int* lab_len_d = label_len.cpu_data();
  const int* seq_len_d = seq_len.cpu_data();

  for (int n = 0; n < N; ++n) {
    // Assume label is in Label proto
    const int label_size = lab_len_d[n];
    // pointer to the first element of the sequence
    const Dtype* label = labels.cpu_data() + n;
    // increment for getting label at next t
    const int label_incr = labels.channels();
    CHECK_GT(label_size, 0)
            << "Labels length is zero in sequence number " << n;

    const int seq_size = seq_len_d[n];  // round Dtype to int for sequence size

    // DLOG(INFO) << "label for sequence number " << n << ": "
    //     << imploded(extract_label_sequence(label, label_size, label_incr));

    // target indices
    std::vector<int> l;

    bool finished_sequence = false;
    const Dtype* prev_label = 0;
    for (int i = 0; i < label_size; ++i) {
      if (i == 0 || !preprocess_collapse_repeated || *label != *prev_label) {
        int i_label = static_cast<int>(*label + 0.5);  // integer label (round)
        if (i_label >= num_classes - 1) {
          finished_sequence = true;
        } else {
          if (finished_sequence) {
            // saw an invalid sequence with non-null following null labels.
            LOG(FATAL) << "Saw a non-null label (index >= num_classes - 1) "
                       << "following a null label, sequence number " << n
                       << ", num_classes " << num_classes << ", labels ["
                       <<  imploded(l) << "]";
          }
          l.push_back(i_label);
        }
      }
      prev_label = label;
      label += label_incr;
    }

    // make sure there is enough time to output the target indices.
    int time = seq_size - output_delay_;
    int required_time = label_size;
    for (int i = 0; i < l.size(); ++i) {
      int l_i = l[i];
      CHECK_GE(l_i, 0) << "All labels must be nonnegative integers. "
                       << "Sequcene number " << n << ", labels "
                       << imploded(l);
      CHECK_LT(l_i, num_classes)
            << "No label may be greater than num_classes: " << num_classes
            << ". At sequence number " << n << ", labels [" << imploded(l)
            << "]";
    }

    CHECK_LE(required_time, time)
          << "Not enough time for target transition sequence";

    // Target indices with blanks before each index and a blank at the end.
    // Length U' = 2U + 1.
    // convert l to l_prime
    GetLPrimeIndices(l, l_primes->at(n));
    *max_u_prime = std::max(*max_u_prime, l_primes->at(n)->count());
  }
}

template <typename Dtype>
void CTCLossLayer<Dtype>::GetLPrimeIndices(const std::vector<int>& l,
                                           SequenceBlob *l_prime) const {
  l_prime->Reshape(2 * l.size() + 1, 1, 1, 1);
  int* l_prime_d = l_prime->mutable_cpu_data();

  for (int i = 0; i < l.size(); ++i) {
    int label = l[i];
    *l_prime_d++ = blank_index_;
    *l_prime_d++ = label;
  }

  *l_prime_d = blank_index_;
}

template <typename Dtype>
void CTCLossLayer<Dtype>::CalculateForwardVariables(
        const SequenceBlob* l_prime,
        const Blob<Dtype>* y,
        bool ctc_merge_repeated,
        CTCVariables* log_alpha) const {
  // Note that the order of log beta is N x T instead of T x N
  const int U = l_prime->count();
  const int T = log_alpha->channels();
  CHECK_EQ(U, log_alpha->num());

  // Data pointers, fill alpha with kLogZero
  double* log_alpha_d = log_alpha->mutable_cpu_data();
  caffe_set(log_alpha->count(), kLogZero, log_alpha->mutable_cpu_data());
  const Dtype* y_d = y->cpu_data();
  const int* l_prime_d = l_prime->cpu_data();

  // Initialize alpha values in Graves Eq (7.5) and Eq (7.6).
  log_alpha_d[log_alpha->offset(0, 0)]
        = log(y_d[y->offset(output_delay_, blank_index_)]);
  // Below, l_prime[1] == label[0]
  const int label_0 = (U > 1) ? l_prime_d[1] : blank_index_;
  log_alpha_d[log_alpha->offset(1, 0)]
        = log(y_d[y->offset(output_delay_, label_0)]);

  for (int t = 1; t < T; ++t) {
    // If there is not enough time to output the remaining labels or
    // some labels have been skipped, then let log_alpha(u, t) continue to
    // be kLogZero.
    for (int u = std::max(0, U - (2 * (T - t)));
         u < std::min(U, 2 * (t + 1));
         ++u) {
      // Begin Graves Eq (7.9)
      // Add in the u, t - 1 term.
      double sum_log_alpha = kLogZero;
      if (ctc_merge_repeated || l_prime_d[u] == blank_index_) {
        sum_log_alpha = log_alpha_d[log_alpha->offset(u, t - 1)];
      }

      // Add in the u - 1, t - 1 term.
      if (u > 0) {
        sum_log_alpha
                = LogSumExp(sum_log_alpha,
                            log_alpha_d[log_alpha->offset(u - 1, t - 1)]);
      }

      // Add in the u - 2, t - 1 term if l_prime(u) != blank or l_prime(u-2).
      if (u > 1) {
        const bool matching_labels_merge
                = ctc_merge_repeated && (l_prime_d[u] == l_prime_d[u - 2]);
        if (l_prime_d[u] != blank_index_ && !matching_labels_merge) {
          sum_log_alpha
                  = LogSumExp(sum_log_alpha,
                              log_alpha_d[log_alpha->offset(u - 2, t - 1)]);
        }
      }
      // Multiply the summed alphas with the activation log probability.
      const Dtype y_v = y_d[y->offset(output_delay_ + t, l_prime_d[u])];
      log_alpha_d[log_alpha->offset(u, t)] = log(y_v) + sum_log_alpha;
    }  // End Graves Eq (7.9)
  }
}

template <typename Dtype>
void CTCLossLayer<Dtype>::CalculateBackwardVariables(
        const SequenceBlob *l_prime,
        const Blob<Dtype>* y,
        bool ctc_merge_repeated,
        CTCVariables *log_beta) const {
  // Note that the order of log beta is N x T instead of T x N
  const int U = l_prime->count();
  const int T = log_beta->channels();
  CHECK_EQ(U, log_beta->num());

  // Data pointers, fill beta with kLogZero
  double *log_beta_d = log_beta->mutable_cpu_data();
  caffe_set(log_beta->count(), kLogZero, log_beta_d);
  const Dtype *y_d = y->cpu_data();

  const int* l_prime_d = l_prime->cpu_data();

  // Initial beta blaues in Graves Eq (7.13): log of probability 1.
  for (int u = U - 2; u < U; ++u) {
    log_beta_d[log_beta->offset(u, T - 1)] = 0;
  }

  for (int t = T - 1 - 1; t >= 0; --t) {
    // If ther is not enough time to output the remaining labels or
    // some labels have been skipped, then let log_beta[u, t] continue to
    // be kLogZero.
    for (int u = std::max(0, U - (2 * (T - t)));
         u < std::min(U, 2 * (t + 1));
         ++u) {
      double &log_beta_ut = log_beta_d[log_beta->offset(u, t)];

      // Begin Graves Eq (7.15)
      // Add in the u, t + 1 term.
      if (ctc_merge_repeated || l_prime_d[u] == blank_index_) {
        const double &log_beta_ut1 = log_beta_d[log_beta->offset(u, t + 1)];
        const double &y_u0
              = y_d[y->offset(output_delay_ + t + 1, l_prime_d[u])];
        DCHECK_GE(y_u0, 0)
              << "Output of the net must be a probability distribution.";
        DCHECK_LE(y_u0, 1)
              << "Output of the net must be a probability distribution.";
        log_beta_ut = LogSumExp(log_beta_ut, log_beta_ut1 + log(y_u0));
      }

      // Add in the u + 1, t + 1 term.
      if (u + 1 < U) {
        const double &log_beta_u1t1
              = log_beta_d[log_beta->offset(u + 1, t + 1)];
        const double &y_u1
              = y_d[y->offset(output_delay_ + t + 1, l_prime_d[u + 1])];
        DCHECK_GE(y_u1, 0)
              << "Output of the net must be a probability distribution.";
        DCHECK_LE(y_u1, 1)
              << "Output of the net must be a probability distribution.";
        log_beta_ut = LogSumExp(log_beta_ut, log_beta_u1t1 + log(y_u1));
      }

      // Add in the u + 2, t + 1 term if l_prime[u] != blank or l_prime[u+2]
      if (u + 2 < U) {
        const bool matching_labels_merge =
                ctc_merge_repeated && (l_prime_d[u] == l_prime_d[u + 2]);
        if (l_prime_d[u] != blank_index_ && !matching_labels_merge) {
          const double &log_beta_u2t1
                = log_beta_d[log_beta->offset(u + 2, t + 1)];
          const double &y_u2
                = y_d[y->offset(output_delay_ + t + 1, l_prime_d[u + 2])];
          DCHECK_GE(y_u2, 0)
                << "Output of the net must be a probability distribution.";
          DCHECK_LE(y_u2, 1)
                << "Output of the net must be a probability distribution.";

          // Add in u + 2 term.
          log_beta_ut = LogSumExp(log_beta_ut, log_beta_u2t1 + log(y_u2));
        }
      }
    }  // End Graves Eq. (7.15)
  }
}

template <typename Dtype>
void CTCLossLayer<Dtype>::CalculateGradient(
        int b,
        int seq_length,
        const SequenceBlob *l_prime,
        const Blob<Dtype> *y_d_blob,
        const CTCVariables* log_alpha,
        const CTCVariables* log_beta,
        double log_p_z_x,
        Blob<Dtype> *y) const {
  const int L = C_;
  const int T = seq_length;
  CHECK_LE(seq_length, y->num());
  CHECK_EQ(L, y->height());
  const int U = l_prime->count();

  const double* log_alpha_d = log_alpha->cpu_data();
  const double* log_beta_d = log_beta->cpu_data();
  const Dtype* y_d = y_d_blob->cpu_data();
  Dtype* y_diff_d = y->mutable_cpu_diff();
  const int* l_prime_d = l_prime->cpu_data();

  DCHECK_EQ(y_diff_d[y->offset(0, b, 0)], static_cast<Dtype>(0));

  // It is possible that no valid path is found if the activations for the
  // targets are zero.
  if (log_p_z_x == kLogZero) {
    LOG(WARNING) << "No valid path found.";
    // dy is then y
    for (int t = 0; t < T - output_delay_; ++t) {
      for (int l = 0; l < L; ++l) {
        y_diff_d[y->offset(output_delay_ + t, b, l)]
              = y_d[y_d_blob->offset(output_delay_ + t, l)];
      }
    }
    return;
  }


  for (int t = 0; t < T - output_delay_; ++t) {
    vector<double> prob_sum(L, kLogZero);

    for (int u = 0; u < U; ++u) {
      const int l = l_prime_d[u];
      prob_sum[l]
            = LogSumExp(prob_sum[l],
                        log_alpha_d[log_alpha->offset(u, t)]
                        + log_beta_d[log_beta->offset(u, t)]);
    }

    for (int l = 0; l < L; ++l) {
      const double negative_term = exp(prob_sum[l] - log_p_z_x);
      y_diff_d[y->offset(output_delay_ + t, b, l)]
              = (y_d[y_d_blob->offset(output_delay_ + t, l)] - negative_term);
    }
  }
}

INSTANTIATE_CLASS(CTCLossLayer);
REGISTER_LAYER_CLASS(CTCLoss);

}  // namespace caffe
