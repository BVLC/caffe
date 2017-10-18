#ifndef CAFFE_CONFUSION_MATRIX_LAYER_HPP_
#define CAFFE_CONFUSION_MATRIX_LAYER_HPP_

#include <functional>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the confusion matrix for a one-of-many
 *        classification task.
 */
  template<typename Dtype>
  class ConfusionMatrixLayer : public Layer<Dtype> {
  public:
    /**
     * @param param provides ConfusionMatrixParameter confusion_matrix_param,
     *     with ConfusionMatrixLayer options:
     */
    explicit ConfusionMatrixLayer(const LayerParameter &param)
      : Layer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                            const vector<Blob<Dtype> *> &top);

    virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                         const vector<Blob<Dtype> *> &top);

    virtual inline const char *type() const { return "ConfusionMatrix"; }

    virtual inline int ExactNumBottomBlobs() const { return 2; }

    // no need to output anything
    virtual inline int MinTopBlobs() const { return 0; }

    virtual inline int MaxTopBlobs() const { return 0; }

    static string GetType();

    virtual void PrintConfusionMatrix(bool reset);

  protected:
    /**
     * @param bottom input Blob vector (length 2)
     *   -# @f$ (N \times C \times H \times W) @f$
     *      the predictions @f$ x @f$, a Blob with values in
     *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
     *      the @f$ K = CHW @f$ classes. Each @f$ x_n @f$ is mapped to a predicted
     *      label @f$ \hat{l}_n @f$ given by its maximal index:
     *      @f$ \hat{l}_n = \arg\max\limits_k x_{nk} @f$
     *   -# @f$ (N \times 1 \times 1 \times 1) @f$
     *      the labels @f$ l @f$, an integer-valued Blob with values
     *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
     *      indicating the correct class label among the @f$ K @f$ classes
     * @param top output Blob vector (length 0 as no need to output anything)
     */
    virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);


    /// @brief Not implemented -- ConfusionMatrixLayer cannot be used as a loss.
    virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool> &propagate_down,
                              const vector<Blob<Dtype> *> &bottom) {
      for (int i = 0; i < propagate_down.size(); ++i) {
        if (propagate_down[i]) { NOT_IMPLEMENTED; }
      }
    }

    int label_axis_, outer_num_, inner_num_;

    /// Whether to ignore instances with a certain label.
    bool has_ignore_label_;
    /// The label indicating that an instance should be ignored.
    int ignore_label_;
    /// Keeps counts of the number of samples per class.
    Blob<Dtype> nums_buffer_;
    /// number of labels
    int num_labels;
    /// current testing iteration
    int current_iter;
    /// number of total testing iteration
    int test_iter;
  };

}  // namespace caffe

#endif  // CAFFE_CONFUSION_MATRIX_LAYER_HPP_
