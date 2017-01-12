/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/util/im_transforms.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

static const float eps = 1e-6;

class ImTransformsTest : public CPUDeviceTest<float> {
};

TEST_F(ImTransformsTest, TestUpdateBBoxByResizePolicy) {
  NormalizedBBox bbox;
  bbox.set_xmin(0.1);
  bbox.set_ymin(0.3);
  bbox.set_xmax(0.3);
  bbox.set_ymax(0.6);
  int img_height = 600;
  int img_width = 1000;
  ResizeParameter resize_param;
  resize_param.set_height(300);
  resize_param.set_width(300);
  NormalizedBBox out_bbox;

  // Test warp.
  out_bbox = bbox;
  resize_param.set_resize_mode(ResizeParameter_Resize_mode_WARP);
  UpdateBBoxByResizePolicy(resize_param, img_width, img_height, &out_bbox);
  EXPECT_NEAR(out_bbox.xmin(), 0.1, eps);
  EXPECT_NEAR(out_bbox.ymin(), 0.3, eps);
  EXPECT_NEAR(out_bbox.xmax(), 0.3, eps);
  EXPECT_NEAR(out_bbox.ymax(), 0.6, eps);

  // Test fit small size.
  out_bbox = bbox;
  resize_param.set_resize_mode(ResizeParameter_Resize_mode_FIT_SMALL_SIZE);
  UpdateBBoxByResizePolicy(resize_param, img_width, img_height, &out_bbox);
  EXPECT_NEAR(out_bbox.xmin(), 0.1, eps);
  EXPECT_NEAR(out_bbox.ymin(), 0.3, eps);
  EXPECT_NEAR(out_bbox.xmax(), 0.3, eps);
  EXPECT_NEAR(out_bbox.ymax(), 0.6, eps);

  // Test fit large size and pad.
  out_bbox = bbox;
  resize_param.set_resize_mode(
      ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD);
  UpdateBBoxByResizePolicy(resize_param, img_width, img_height, &out_bbox);
  EXPECT_NEAR(out_bbox.xmin(), 0.1, eps);
  EXPECT_NEAR(out_bbox.ymin(), (180 * 0.3 + 60) / 300, eps);
  EXPECT_NEAR(out_bbox.xmax(), 0.3, eps);
  EXPECT_NEAR(out_bbox.ymax(), (180 * 0.6 + 60) / 300, eps);

  /*** Reverse the image size. ***/
  img_height = 1000;
  img_width = 600;

  // Test fit large size and pad.
  out_bbox = bbox;
  resize_param.set_resize_mode(
      ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD);
  UpdateBBoxByResizePolicy(resize_param, img_width, img_height, &out_bbox);
  EXPECT_NEAR(out_bbox.xmin(), (180 * 0.1 + 60) / 300, eps);
  EXPECT_NEAR(out_bbox.ymin(), 0.3, eps);
  EXPECT_NEAR(out_bbox.xmax(), (180 * 0.3 + 60) / 300, eps);
  EXPECT_NEAR(out_bbox.ymax(), 0.6, eps);
}

#ifdef USE_OPENCV
TEST_F(ImTransformsTest, TestApplyResize) {
  cv::Mat in_img(60, 100, CV_8UC3);
  cv::Mat out_img;
  ResizeParameter resize_param;
  resize_param.set_height(30);
  resize_param.set_width(30);

  resize_param.set_resize_mode(ResizeParameter_Resize_mode_WARP);
  out_img = ApplyResize(in_img, resize_param);
  CHECK_EQ(out_img.cols, 30);
  CHECK_EQ(out_img.rows, 30);

  resize_param.set_resize_mode(ResizeParameter_Resize_mode_FIT_SMALL_SIZE);
  out_img = ApplyResize(in_img, resize_param);
  CHECK_EQ(out_img.cols, 50);
  CHECK_EQ(out_img.rows, 30);

  resize_param.set_resize_mode(
      ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD);
  out_img = ApplyResize(in_img, resize_param);
  CHECK_EQ(out_img.cols, 30);
  CHECK_EQ(out_img.rows, 30);
}
#endif  // USE_OPENCV

}  // namespace caffe
