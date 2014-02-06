// Copyright 2013 Yangqing Jia

#include <cstring>
#include <algorithm>

#include "gtest/gtest.h"
#include "caffe/common.hpp"

#include "caffe/proto/caffe.pb.h"

#include "caffe/test/test_caffe_main.hpp"

#include "caffe/net.hpp"
#include "caffe/solver.hpp"


namespace caffe {

  typedef double Dtype;

  TEST(TestTerminationCriterion, MaxIter) {
    MaxIterTerminationCriterion<Dtype> criterion(3);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyIteration(1);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyIteration(2);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyIteration(3);
    EXPECT_TRUE(criterion.IsCriterionMet());
  }
  
  TEST(TestTerminationCriterion, TestAccuracy) {
    TestAccuracyTerminationCriterion<Dtype> criterion(3);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.NotifyTestAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());

    //first countdown
    criterion.NotifyTestAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //second countdown
    criterion.NotifyTestAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //reset
    criterion.NotifyTestAccuracy(0.6);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //first countdown
    criterion.NotifyTestAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //second countdown
    criterion.NotifyTestAccuracy(0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //third countdown
    criterion.NotifyTestAccuracy(0.5);
    
    EXPECT_TRUE(criterion.IsCriterionMet());
  }

}  // namespace caffe
