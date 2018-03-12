#ifndef CAFFE_CTC_LOSS_LAYER_HPP
#define CAFFE_CTC_LOSS_LAYER_HPP

#include <list>
#include <vector>

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
template <typename Dtype>


/**
 * @brief Implementation of the CTC (Connectionist Temporal Classification) algorithm
 *        to label unsegmented sequence data with recurrent neural networks
 *
 * The input data is expected to follow the rules for the recurrent layers, meaning:
 * T x N x L, where T is the time compontent, N the number of simulaneously computed
 * input sequences and L is the size of the possible labels. There will be a softmax
 * applied to the input data. No need to add a manual softmax layer. Note that L
 * must be the size of your actual label count plus one. This last entry represents
 * the required 'blank_index' for the algorith.
 *
 * The second input blob are the sequence indicators for the data with shape T x N.
 * A 0 means the start of a sequence. See RecurrentLayer for additional information.
 *
 * The third input is the blob of the target sequence with shape T X N. The data
 * is expected to contain the labeling of the target sequence and -1 if the sequence
 * has ended.
 *
 * Sample input data for T = 10, N = 1 (this column is dropped in the data), C = 5
 * (the data is filled with dummy values), and the target sequence [012].
 * The input sequence has a length of 8. The number of labels is 4 (3 + 1).
 *
 * T  | data        | seq_ind | target_seq |
 * -- | ----------- | ------- | ---------- |
 * 0  | [1 5 2 2 5] |    0    |     0      |
 * 1  | [3 3 2 3 4] |    1    |     2      |
 * 2  | [7 3 3 5 4] |    1    |     1      |
 * 3  | [0 5 3 2 5] |    1    |    -1      |
 * 4  | [0 4 1 2 4] |    1    |    -1      |
 * 5  | [2 4 3 5 7] |    1    |    -1      |
 * 6  | [3 4 1 3 4] |    1    |    -1      |
 * 7  | [8 4 2 4 4] |    1    |    -1      |
 * 8  | [0 0 0 0 0] |    0    |    -1      |
 * 9  | [0 0 0 0 0] |    0    |    -1      |
 *
 * Note that the complete sequence must fit into a (time) batch and each sequence
 * must start at t = 0 of that batch.
 *
 * To split the computation into Forward and Backward passes the intermediate results
 * (alpha, beta, l_prime, log_p_z_x) are stored during the forward pass and are
 * reused during the backward pass.
 */
class CTCLossLayer : public LossLayer<Dtype> {
 public:
  // double blob for storing probabilities with higher accuracy
  typedef Blob<double> ProbBlob;

  // alpha or beta variables are a probability blob
  typedef ProbBlob CTCVariables;

  // blob for storing lengths (sequences)
  typedef Blob<int> LengthBlob;

  // blob for storing a sequence
  typedef Blob<int> SequenceBlob;

  // Vector storing the label sequences for each sequence
  typedef vector<SequenceBlob*> LabelSequences;

 public:
  explicit CTCLossLayer(const LayerParameter& param);
  virtual ~CTCLossLayer();

  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CTCLoss"; }

  // probabilities, sequence indicators, target sequence
  virtual inline int ExactNumBottomBlobs() const { return 3; }

  // loss
  virtual inline int ExactNumTopBlobs() const { return 1; }

  // access to internal calculation variables,
  // used for testing intermediate states
  void SetLossCalculationT(int t) {loss_calculation_t_ = t;}
  const LengthBlob& SequenceLength() const {return seq_len_;}
  const LengthBlob& LabelLength() const {return label_len_;}
  const ProbBlob& LogPzx() const {return log_p_z_x_;}
  const vector<CTCVariables*>& LogAlpha() const {return log_alpha_;}
  const vector<CTCVariables*>& LogBeta() const {return log_beta_;}
  const LabelSequences& LPrimes() const {return l_primes_;}
  const vector<Blob<Dtype>*>& Y() const {return y_;}

 protected:
  /**
   * @brief Computes the loss and the error gradients for the input data
   *        in one step (due to optimization isses)
   *
   * @param bottom input Blob vector (length 3)
   *   -# @f$ (T \times N \times C) @f$
   *      the inputs @f$ x @f$
   *   -# @f$ (T \times N) @f$
   *      the sequence indicators for the data
   *      (must be 0 at @f$ t = 0 @f$ and 1 during a sequence)
   *   -# @f$ (T \times N) @f$
   *      the target sequence
   *      (must start at @f$ t = 0 @f$ and filled with -1 if the sequence has ended)
   * @param top output Blob vector (length 1)
   *   -# @f$ (1) @f$
   *      the computed loss
   */

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  /**
   * @brief Unused. Gradient calculation is done in Forward_cpu
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

 private:
  /**
   * @brief Calculates the loss of the given data batch
   * @param seq_len_blob The blob containing the data sequence lengths
   * @param target_seq_blob The blob containing the target sequence
   * @param target_seq_len_blob The blob containing the target sequence lengths
   * @param data_blob The blob containing the probability distribution
   * @param preprocess_collapse_repeated See preprocess_collapse_repeated_
   * @param ctc_merge_repeated See ctc_merge_repeated_
   * @param loss The output of the loss
   * @param requires_backprob Calculate the gradients
   */
  void CalculateLoss(const LengthBlob *seq_len_blob,
                     const Blob<Dtype> *target_seq_blob,
                     const LengthBlob *target_seq_len_blob,
                     Blob<Dtype> *data_blob,
                     ProbBlob *log_p_z_x,
                     LabelSequences *l_primes,
                     vector<Blob<Dtype>*> *y,
                     bool preprocess_collapse_repeated,
                     bool ctc_merge_repeated,
                     Dtype *loss,
                     bool requires_backprob) const;

  /**
   * @brief Calculates the forward variables of the CTC algorithm denoted by
   *        @f$ \alpha @f$
   * @param l_prime The target sequence with inserted blanks
   * @param y The input probabilities for this sequence
   * @param ctc_merge_repeated See ctc_merge_repeated_
   * @param log_alpha The output blob that stores the variables
   */
  void CalculateForwardVariables(const SequenceBlob* l_prime,
                                 const Blob<Dtype>* y,
                                 bool ctc_merge_repeated,
                                 CTCVariables* log_alpha) const;

  /**
   * @brief Calculates the backward variables of the CTC algorithm denoted by
   *        @f$ \beta @f$
   * @param l_prime The target sequence with inserted blanks
   * @param y The input probabilities for this sequence
   * @param ctc_merge_repeated See ctc_merge_repeated_
   * @param log_beta The output blob that stores the variables
   */
  void CalculateBackwardVariables(const SequenceBlob* l_prime,
                                  const Blob<Dtype>* y,
                                  bool ctc_merge_repeated,
                                  CTCVariables* log_beta) const;

  /**
   * @brief Calculate the gradient of the input variables
   * @param b The number of the sequence in the input data
   * @param seq_length The sequence length
   * @param l_prime The target sequence with inserted blanks
   * @param y_d_blob The softmax input variables for this specific sequence
   * @param log_alpha The log values of the forward variables
   * @param log_beta The log values of the backward variables
   * @param log_p_z_x The computed probability of the path (corresponds to loss)
   * @param y The input blob for this sequence. Here only the diff data will be used
   *          as output for the gradient.
   */
  void CalculateGradient(int b, int seq_length, const SequenceBlob* l_prime,
                         const Blob<Dtype>* y_d_blob,
                         const CTCVariables* log_alpha,
                         const CTCVariables* log_beta,
                         double log_p_z_x,
                         Blob<Dtype>* y) const;

  /**
   * @brief Computes the L' sequence of the target sequences
   * @param preprocess_collapse_repeated See proprocess_collapse_repeated_
   * @param N The number of parallel sequences
   * @param num_classes The number of allowed labels
   * @param seq_len The sequence lengths (lengths of the input data)
   * @param labels The labels blob
   * @param label_len The length of the labels for each sequence
   * @param max_u_prime Output of the maximum length of a target sequence
   * @param l_primes Output of the label sequences
   */
  void PopulateLPrimes(bool preprocess_collapse_repeated,
                       int N,
                       int num_classes,
                       const LengthBlob& seq_len,
                       const Blob<Dtype>& labels,
                       const LengthBlob& label_len,
                       int *max_u_prime,
                       LabelSequences* l_primes) const;
  /**
   * @brief Transform a sequence to the sequence with inserted blanks.
   * @param l the default sequence
   * @param l_prime the sequence with inserted blanks
   *
   * The length of the output will be |L'| = 2 |L| + 1.
   *
   * E.g. [0 1 4] -> [5 0 5 1 5 4 5] where 5 indicates the blank label.
   * The number of classes is therefore 6
   */
  void GetLPrimeIndices(const vector<int>& l, SequenceBlob* l_prime) const;

  int T_;
  int N_;
  int C_;

  int output_delay_;
  int blank_index_;

  bool preprocess_collapse_repeated_;
  bool ctc_merge_repeated_;

  /// the time for which to calculate the loss
  /// see Graves Eq. (7.27)
  /// Note that the result must be the same for each 0 <= t < T
  /// Therefore you can chose an arbitrary value, default 0
  int loss_calculation_t_;

  // Intermediate variables that are calculated during the forward pass
  // and reused during the backward pass

  // blob to store the sequence lengths (input data)
  LengthBlob seq_len_;

  // blob to store the label lengths (target label sequence)
  LengthBlob label_len_;

  // blob to store log(p(z|x)) for each batch
  ProbBlob log_p_z_x_;

  // blobs to store the alpha and beta variables of each input sequence
  // the algorithm will store the logarithm of these variables
  vector<CTCVariables*> log_alpha_;
  vector<CTCVariables*> log_beta_;

  // blobs to store the l_primes of the sequences
  LabelSequences l_primes_;

  // blobs to store the intermediate softmax outputs
  vector<Blob<Dtype>*> y_;
};

}  // namespace caffe

#endif  // CAFFE_CTC_LOSS_LAYER_HPP
