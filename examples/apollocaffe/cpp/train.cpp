#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include "caffe/apollonet.hpp"
#include "caffe/caffe.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;     // NOLINT(build/namespaces)

using std::vector;

int main() {
  ApolloNet<float> net;
  shared_ptr<Layer<float> > dataLayer, labelLayer;
  shared_ptr<MatDataLayer<float> > dataMemLayer, labelMemLayer;
  typedef LayerRegistry<float> LR;
  dataLayer = LR::CreateLayer("name: 'data' type: 'MatData'"
    "top: 'data' phase: TEST");
  labelLayer = LR::CreateLayer("name: 'label' type: 'MatData'"
    "top: 'label' phase: TEST");
  dataMemLayer = boost::dynamic_pointer_cast<MatDataLayer<float> > (dataLayer);
  labelMemLayer =
    boost::dynamic_pointer_cast<MatDataLayer<float> > (labelLayer);
  Mat example = Mat(1, 1, CV_8UC1);
  vector<Mat> matVec(1);
  vector<Mat> labelMatVec(1);
  double loss;
  for (int i = 0; i < 200; ++i) {
    randu(example, Scalar::all(0), Scalar::all(50));
    matVec[0] = example;
    labelMatVec[0] = example * 3;
    dataMemLayer->AddMatVector(matVec);
    labelMemLayer->AddMatVector(labelMatVec);
    net.f(dataLayer);
    net.f(labelLayer);
    net.f("name: 'conv' type: 'Convolution' bottom: 'data' top: 'conv'"
        "convolution_param { num_output: 1 weight_filler { type: 'xavier' } "
        "bias_filler { type: 'constant' value: 0.0 } kernel_h: 1 kernel_w: 1}");
    loss = net.f("name: 'loss' type: 'EuclideanLoss' bottom: 'conv' "
        "bottom: 'label' top: 'loss' ");
    net.Backward();
    net.Update(0.0001, 0, -1, 0);
    net.ResetForward();
    if (i % 10 == 0) {
      std::cout << loss << std::endl;
    }
  }
  return 0;
}
