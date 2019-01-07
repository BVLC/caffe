#ifndef CAFFE_GAN_SOLVER_HPP_
#define CAFFE_GAN_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype> class Solver;


/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on GAN%s.
 *
 * Consist of two solver for generator and discriminator individually.
 */
template <typename Dtype>
class GANSolver {
 public:
  explicit GANSolver(const SolverParameter& g_param, const SolverParameter& d_param);

  shared_ptr<caffe::Solver<Dtype> > getDiscriminatorSolver() {return d_solver;}
  shared_ptr<caffe::Solver<Dtype> > getGeneratorSolver() {return g_solver;}

  void SetActionFunction(ActionCallback func);

  void Restore(const char* resume_file) {
    // TODO
  }

  void tile(const vector<cv::Mat> &src, cv::Mat &dst, int grid_x, int grid_y) {
    // patch size
    int width  = dst.cols/grid_x;
    int height = dst.rows/grid_y;
    // iterate through grid
    int k = 0;
    for(int i = 0; i < grid_y; i++) {
      for(int j = 0; j < grid_x; j++) {
        cv::Mat s = src[k++];
        cv::resize(s,s,cv::Size(width,height));
        s.copyTo(dst(cv::Rect(j*width,i*height,width,height)));
      }
    }
  }

  /// Show 3 channel or 1 channel blob; batch size >= 16; scale between (-1, 1)
  cv::Mat* blob2cvgrid(Blob<Dtype> *blob) {
    // LOG(INFO) << "Shape " << blob->shape_string();
    int width = blob->width(), height = blob->height(), channels = blob->channels();
    Dtype* input_data = blob->mutable_cpu_data();
    vector<cv::Mat> src;
    for (int i = 0; i < 16; i ++) {
      cv::Mat image;
      vector<cv::Mat> color_channel;
      for(int i = 0; i < channels; i ++) {
        cv::Mat _ch(height, width, CV_32FC1, input_data);
        color_channel.push_back(_ch);
        input_data += height * width;
      }
      cv::merge(color_channel, image);
      src.push_back(image);
    }
    cv::Mat *grid = new cv::Mat(height * 4, width * 4, (channels == 1 ? CV_32FC1: CV_32FC3));
    tile(src, *grid, 4, 4);

    //double min, max;
    //cv::minMaxLoc(*grid, &min, &max);
    //LOG(INFO) << "Min " <<  min << " Max " << max;

    *grid = (*grid + 1) * 127.5; 
    return grid;
  }

  void TestAll() {
    // Must be float
    cv::Mat *x_fake_grid = blob2cvgrid(g_solver->net_->output_blobs()[0]);
    
    string name = d_solver->param_.snapshot_prefix() + "x_fake_" + caffe::format_int(iter_) + ".png";
    cv::imwrite(name.c_str(), *x_fake_grid);
    delete x_fake_grid;

    d_solver->net_->Forward();
    int ind = d_solver->net_->base_layer_index();

    auto vecs = d_solver->net_->bottom_vecs();

    cv::Mat *x_real_grid = blob2cvgrid(vecs[ind][0]);
    name = d_solver->param_.snapshot_prefix() + "x_real_" + caffe::format_int(iter_) + ".png";
    cv::imwrite(name.c_str(), *x_real_grid);
    delete x_real_grid;
  }

  SolverAction::Enum GetRequestedAction();
  
  void Step(int iters);

  virtual void Solve(const char* resume_file = NULL);

 private:
  shared_ptr<caffe::Solver<Dtype> > g_solver, d_solver;

  int iter_;
  int current_step_;

  vector<Dtype> d_losses_, g_losses_;
  Dtype d_smoothed_loss_, g_smoothed_loss_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  // Timing information, handy to tune e.g. nbr of GPUs
  Timer iteration_timer_;
  float iterations_last_;

  DISABLE_COPY_AND_ASSIGN(GANSolver);
};

}  // namespace caffe

#endif  // CAFFE_GAN_SOLVER_HPP_
