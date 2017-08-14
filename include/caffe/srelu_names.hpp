#ifndef CAFFE_SRELU_NAMES_HPP_
#define CAFFE_SRELU_NAMES_HPP_

using std::vector;
using std::string;

vector<string> GOOGLENET_BLOBS={
   "conv1/7x7_s2", "conv2/3x3_reduce", "conv2/3x3",
   "inception_3a/1x1", "inception_3a/3x3_reduce", "inception_3a/3x3",
       "inception_3a/5x5_reduce", "inception_3a/5x5", "inception_3a/pool_proj",
   "inception_3b/1x1", "inception_3b/3x3_reduce", "inception_3b/3x3",
       "inception_3b/5x5_reduce", "inception_3b/5x5", "inception_3b/pool_proj",
   "inception_4a/1x1", "inception_4a/3x3_reduce", "inception_4a/3x3",
       "inception_4a/5x5_reduce", "inception_4a/5x5", "inception_4a/pool_proj",
   "inception_4b/1x1", "inception_4b/3x3_reduce", "inception_4b/3x3",
       "inception_4b/5x5_reduce", "inception_4b/5x5", "inception_4b/pool_proj",
   "inception_4c/1x1", "inception_4c/3x3_reduce", "inception_4c/3x3",
       "inception_4c/5x5_reduce", "inception_4c/5x5", "inception_4c/pool_proj",
   "inception_4d/1x1", "inception_4d/3x3_reduce", "inception_4d/3x3",
       "inception_4d/5x5_reduce", "inception_4d/5x5", "inception_4d/pool_proj",
   "inception_4e/1x1", "inception_4e/3x3_reduce", "inception_4e/3x3",
       "inception_4e/5x5_reduce", "inception_4e/5x5", "inception_4e/pool_proj",
   "inception_5a/1x1", "inception_5a/3x3_reduce", "inception_5a/3x3",
       "inception_5a/5x5_reduce", "inception_5a/5x5", "inception_5a/pool_proj",
   "inception_5b/1x1", "inception_5b/3x3_reduce", "inception_5b/3x3",
       "inception_5b/5x5_reduce", "inception_5b/5x5", "inception_5b/pool_proj"
};

vector<string> LAYER_NAMES={
    "conv1", "cccp1", "cccp2",
    "conv2", "cccp3", "cccp4",
    "conv3", "cccp5", "cccp6"
};

vector<string> SRELU_NAMES={
    "conv1/relu_7x7_thresh",                  "conv1/relu_7x7_pslope",                  "conv1/relu_7x7_nslope",                  "conv1/relu_7x7_nthresh",
    "conv2/relu_3x3_reduce_thresh",           "conv2/relu_3x3_reduce_pslope",           "conv2/relu_3x3_reduce_nslope",           "conv2/relu_3x3_reduce_nthresh",
    "conv2/relu_3x3_thresh",                  "conv2/relu_3x3_pslope",                  "conv2/relu_3x3_nslope",                  "conv2/relu_3x3_nthresh",

    "inception_3a/relu_1x1_thresh",           "inception_3a/relu_1x1_pslope",           "inception_3a/relu_1x1_nslope",           "inception_3a/relu_1x1_nthresh",
    "inception_3a/relu_3x3_reduce_thresh",    "inception_3a/relu_3x3_reduce_pslope",    "inception_3a/relu_3x3_reduce_nslope",    "inception_3a/relu_3x3_reduce_nthresh",
    "inception_3a/relu_3x3_thresh",           "inception_3a/relu_3x3_pslope",           "inception_3a/relu_3x3_nslope",           "inception_3a/relu_3x3_nthresh",
    "inception_3a/relu_5x5_reduce_thresh",    "inception_3a/relu_5x5_reduce_pslope",    "inception_3a/relu_5x5_reduce_nslope",    "inception_3a/relu_5x5_reduce_nthresh",
    "inception_3a/relu_5x5_thresh",           "inception_3a/relu_5x5_pslope",           "inception_3a/relu_5x5_nslope",           "inception_3a/relu_5x5_nthresh",
    "inception_3a/relu_pool_proj_thresh",     "inception_3a/relu_pool_proj_pslope",     "inception_3a/relu_pool_proj_nslope",     "inception_3a/relu_pool_proj_nthresh",
    "inception_3b/relu_1x1_thresh",           "inception_3b/relu_1x1_pslope",           "inception_3b/relu_1x1_nslope",           "inception_3b/relu_1x1_nthresh",
    "inception_3b/relu_3x3_reduce_thresh",    "inception_3b/relu_3x3_reduce_pslope",    "inception_3b/relu_3x3_reduce_nslope",    "inception_3b/relu_3x3_reduce_nthresh",
    "inception_3b/relu_3x3_thresh",           "inception_3b/relu_3x3_pslope",           "inception_3b/relu_3x3_nslope",           "inception_3b/relu_3x3_nthresh",
    "inception_3b/relu_5x5_reduce_thresh",    "inception_3b/relu_5x5_reduce_pslope",    "inception_3b/relu_5x5_reduce_nslope",    "inception_3b/relu_5x5_reduce_nthresh",
    "inception_3b/relu_5x5_thresh",           "inception_3b/relu_5x5_pslope",           "inception_3b/relu_5x5_nslope",           "inception_3b/relu_5x5_nthresh",
    "inception_3b/relu_pool_proj_thresh",     "inception_3b/relu_pool_proj_pslope",     "inception_3b/relu_pool_proj_nslope",     "inception_3b/relu_pool_proj_nthresh",
    "inception_4a/relu_1x1_thresh",           "inception_4a/relu_1x1_pslope",           "inception_4a/relu_1x1_nslope",           "inception_4a/relu_1x1_nthresh",
    "inception_4a/relu_3x3_reduce_thresh",    "inception_4a/relu_3x3_reduce_pslope",    "inception_4a/relu_3x3_reduce_nslope",    "inception_4a/relu_3x3_reduce_nthresh",
    "inception_4a/relu_3x3_thresh",           "inception_4a/relu_3x3_pslope",           "inception_4a/relu_3x3_nslope",           "inception_4a/relu_3x3_nthresh",
    "inception_4a/relu_5x5_reduce_thresh",    "inception_4a/relu_5x5_reduce_pslope",    "inception_4a/relu_5x5_reduce_nslope",    "inception_4a/relu_5x5_reduce_nthresh",
    "inception_4a/relu_5x5_thresh",           "inception_4a/relu_5x5_pslope",           "inception_4a/relu_5x5_nslope",           "inception_4a/relu_5x5_nthresh",
    "inception_4a/relu_pool_proj_thresh",     "inception_4a/relu_pool_proj_pslope",     "inception_4a/relu_pool_proj_nslope",     "inception_4a/relu_pool_proj_nthresh",

    "loss1/relu_conv_thresh",                 "loss1/relu_conv_pslope",                 "loss1/relu_conv_nslope",                 "loss1/relu_conv_nthresh",
    "loss1/relu_fc_thresh",                   "loss1/relu_fc_pslope",                   "loss1/relu_fc_nslope",                   "loss1/relu_fc_nthresh",

    "inception_4b/relu_1x1_thresh",           "inception_4b/relu_1x1_pslope",           "inception_4b/relu_1x1_nslope",           "inception_4b/relu_1x1_nthresh",
    "inception_4b/relu_3x3_reduce_thresh",    "inception_4b/relu_3x3_reduce_pslope",    "inception_4b/relu_3x3_reduce_nslope",    "inception_4b/relu_3x3_reduce_nthresh",
    "inception_4b/relu_3x3_thresh",           "inception_4b/relu_3x3_pslope",           "inception_4b/relu_3x3_nslope",           "inception_4b/relu_3x3_nthresh",
    "inception_4b/relu_5x5_reduce_thresh",    "inception_4b/relu_5x5_reduce_pslope",    "inception_4b/relu_5x5_reduce_nslope",    "inception_4b/relu_5x5_reduce_nthresh",
    "inception_4b/relu_5x5_thresh",           "inception_4b/relu_5x5_pslope",           "inception_4b/relu_5x5_nslope",           "inception_4b/relu_5x5_nthresh",
    "inception_4b/relu_pool_proj_thresh",     "inception_4b/relu_pool_proj_pslope",     "inception_4b/relu_pool_proj_nslope",     "inception_4b/relu_pool_proj_nthresh",
    "inception_4c/relu_1x1_thresh",           "inception_4c/relu_1x1_pslope",           "inception_4c/relu_1x1_nslope",           "inception_4c/relu_1x1_nthresh",
    "inception_4c/relu_3x3_reduce_thresh",    "inception_4c/relu_3x3_reduce_pslope",    "inception_4c/relu_3x3_reduce_nslope",    "inception_4c/relu_3x3_reduce_nthresh",
    "inception_4c/relu_3x3_thresh",           "inception_4c/relu_3x3_pslope",           "inception_4c/relu_3x3_nslope",           "inception_4c/relu_3x3_nthresh",
    "inception_4c/relu_5x5_reduce_thresh",    "inception_4c/relu_5x5_reduce_pslope",    "inception_4c/relu_5x5_reduce_nslope",    "inception_4c/relu_5x5_reduce_nthresh",
    "inception_4c/relu_5x5_thresh",           "inception_4c/relu_5x5_pslope",           "inception_4c/relu_5x5_nslope",           "inception_4c/relu_5x5_nthresh",
    "inception_4c/relu_pool_proj_thresh",     "inception_4c/relu_pool_proj_pslope",     "inception_4c/relu_pool_proj_nslope",     "inception_4c/relu_pool_proj_nthresh",
    "inception_4d/relu_1x1_thresh",           "inception_4d/relu_1x1_pslope",           "inception_4d/relu_1x1_nslope",           "inception_4d/relu_1x1_nthresh",
    "inception_4d/relu_3x3_reduce_thresh",    "inception_4d/relu_3x3_reduce_pslope",    "inception_4d/relu_3x3_reduce_nslope",    "inception_4d/relu_3x3_reduce_nthresh",
    "inception_4d/relu_3x3_thresh",           "inception_4d/relu_3x3_pslope",           "inception_4d/relu_3x3_nslope",           "inception_4d/relu_3x3_nthresh",
    "inception_4d/relu_5x5_reduce_thresh",    "inception_4d/relu_5x5_reduce_pslope",    "inception_4d/relu_5x5_reduce_nslope",    "inception_4d/relu_5x5_reduce_nthresh",
    "inception_4d/relu_5x5_thresh",           "inception_4d/relu_5x5_pslope",           "inception_4d/relu_5x5_nslope",           "inception_4d/relu_5x5_nthresh",
    "inception_4d/relu_pool_proj_thresh",     "inception_4d/relu_pool_proj_pslope",     "inception_4d/relu_pool_proj_nslope",     "inception_4d/relu_pool_proj_nthresh",

    "loss2/relu_conv_thresh",                 "loss2/relu_conv_pslope",                 "loss2/relu_conv_nslope",                 "loss2/relu_conv_nthresh",
    "loss2/relu_fc_thresh",                   "loss2/relu_fc_pslope",                   "loss2/relu_fc_nslope",                   "loss2/relu_fc_nthresh",

    "inception_4e/relu_1x1_thresh",           "inception_4e/relu_1x1_pslope",           "inception_4e/relu_1x1_nslope",           "inception_4e/relu_1x1_nthresh",
    "inception_4e/relu_3x3_reduce_thresh",    "inception_4e/relu_3x3_reduce_pslope",    "inception_4e/relu_3x3_reduce_nslope",    "inception_4e/relu_3x3_reduce_nthresh",
    "inception_4e/relu_3x3_thresh",           "inception_4e/relu_3x3_pslope",           "inception_4e/relu_3x3_nslope",           "inception_4e/relu_3x3_nthresh",
    "inception_4e/relu_5x5_reduce_thresh",    "inception_4e/relu_5x5_reduce_pslope",    "inception_4e/relu_5x5_reduce_nslope",    "inception_4e/relu_5x5_reduce_nthresh",
    "inception_4e/relu_5x5_thresh",           "inception_4e/relu_5x5_pslope",           "inception_4e/relu_5x5_nslope",           "inception_4e/relu_5x5_nthresh",
    "inception_4e/relu_pool_proj_thresh",     "inception_4e/relu_pool_proj_pslope",     "inception_4e/relu_pool_proj_nslope",     "inception_4e/relu_pool_proj_nthresh",
    "inception_5a/relu_1x1_thresh",           "inception_5a/relu_1x1_pslope",           "inception_5a/relu_1x1_nslope",           "inception_5a/relu_1x1_nthresh",
    "inception_5a/relu_3x3_reduce_thresh",    "inception_5a/relu_3x3_reduce_pslope",    "inception_5a/relu_3x3_reduce_nslope",    "inception_5a/relu_3x3_reduce_nthresh",
    "inception_5a/relu_3x3_thresh",           "inception_5a/relu_3x3_pslope",           "inception_5a/relu_3x3_nslope",           "inception_5a/relu_3x3_nthresh",
    "inception_5a/relu_5x5_reduce_thresh",    "inception_5a/relu_5x5_reduce_pslope",    "inception_5a/relu_5x5_reduce_nslope",    "inception_5a/relu_5x5_reduce_nthresh",
    "inception_5a/relu_5x5_thresh",           "inception_5a/relu_5x5_pslope",           "inception_5a/relu_5x5_nslope",           "inception_5a/relu_5x5_nthresh",
    "inception_5a/relu_pool_proj_thresh",     "inception_5a/relu_pool_proj_pslope",     "inception_5a/relu_pool_proj_nslope",     "inception_5a/relu_pool_proj_nthresh",
    "inception_5b/relu_1x1_thresh",           "inception_5b/relu_1x1_pslope",           "inception_5b/relu_1x1_nslope",           "inception_5b/relu_1x1_nthresh",
    "inception_5b/relu_3x3_reduce_thresh",    "inception_5b/relu_3x3_reduce_pslope",    "inception_5b/relu_3x3_reduce_nslope",    "inception_5b/relu_3x3_reduce_nthresh",
    "inception_5b/relu_3x3_thresh",           "inception_5b/relu_3x3_pslope",           "inception_5b/relu_3x3_nslope",           "inception_5b/relu_3x3_nthresh",
    "inception_5b/relu_5x5_reduce_thresh",    "inception_5b/relu_5x5_reduce_pslope",    "inception_5b/relu_5x5_reduce_nslope",    "inception_5b/relu_5x5_reduce_nthresh",
    "inception_5b/relu_5x5_thresh",           "inception_5b/relu_5x5_pslope",           "inception_5b/relu_5x5_nslope",           "inception_5b/relu_5x5_nthresh",
    "inception_5b/relu_pool_proj_thresh",     "inception_5b/relu_pool_proj_pslope",     "inception_5b/relu_pool_proj_nslope",     "inception_5b/relu_pool_proj_nthresh"
};

// vector<string> SRELU_NAMES={
//     "srelu1_thresh",       "srelu1_pslope",       "srelu1_nslope",       "srelu1_nthresh",
//     "srelu_cccp1_thresh",  "srelu_cccp1_pslope",  "srelu_cccp1_nslope",  "srelu_cccp1_nthresh",
//     "srelu_cccp2_thresh",  "srelu_cccp2_pslope",  "srelu_cccp2_nslope",  "srelu_cccp2_nthresh",
//     "srelu2_thresh",       "srelu2_pslope",       "srelu2_nslope",       "srelu2_nthresh",
//     "srelu_cccp3_thresh",  "srelu_cccp3_pslope",  "srelu_cccp3_nslope",  "srelu_cccp3_nthresh",
//     "srelu_cccp4_thresh",  "srelu_cccp4_pslope",  "srelu_cccp4_nslope",  "srelu_cccp4_nthresh",
//     "srelu3_thresh",       "srelu3_pslope",       "srelu3_nslope",       "srelu3_nthresh",
//     "srelu_cccp5_thresh",  "srelu_cccp5_pslope",  "srelu_cccp5_nslope",  "srelu_cccp5_nthresh",
//     "srelu_cccp6_thresh",  "srelu_cccp6_pslope",  "srelu_cccp6_nslope",  "srelu_cccp6_nthresh"
// };

vector<string> SRELU_THRESH_NAMES={
    "srelu1_thresh",
    "srelu_cccp1_thresh",
    "srelu_cccp2_thresh",
    "srelu2_thresh",
    "srelu_cccp3_thresh",
    "srelu_cccp4_thresh",
    "srelu3_thresh",
    "srelu_cccp5_thresh",
    "srelu_cccp6_thresh",
};

vector<string> SRELU_PSLOPE_NAMES={
    "srelu1_pslope",
    "srelu_cccp1_pslope",
    "srelu_cccp2_pslope",
    "srelu2_pslope",
    "srelu_cccp3_pslope",
    "srelu_cccp4_pslope",
    "srelu3_pslope",
    "srelu_cccp5_pslope",
    "srelu_cccp6_pslope"
};

vector<string> SRELU_NSLOPE_NAMES={
  "srelu1_nslope",
  "srelu_cccp1_nslope",
  "srelu_cccp2_nslope",
  "srelu2_nslope",
  "srelu_cccp3_nslope",
  "srelu_cccp4_nslope",
  "srelu3_nslope",
  "srelu_cccp5_nslope",
  "srelu_cccp6_nslope"
};

vector<string> SRELU_NTHRESH_NAMES={
  "srelu1_nthresh",
  "srelu_cccp1_nthresh",
  "srelu_cccp2_nthresh",
  "srelu2_nthresh",
  "srelu_cccp3_nthresh",
  "srelu_cccp4_nthresh",
  "srelu3_nthresh",
  "srelu_cccp5_nthresh",
  "srelu_cccp6_nthresh"
};

#endif
