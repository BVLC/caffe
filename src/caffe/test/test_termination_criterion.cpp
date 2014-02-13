// Copyright 2014 Tobias Domhan

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
    
    criterion.Notify(TerminationCriterionBase::TYPE_ITERATION, 1);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.Notify(TerminationCriterionBase::TYPE_ITERATION, 2);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.Notify(TerminationCriterionBase::TYPE_ITERATION, 3);
    EXPECT_TRUE(criterion.IsCriterionMet());
  }
  
  TEST(TestTerminationCriterion, TestAccuracy) {
    TestAccuracyTerminationCriterion<Dtype> criterion(3);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    criterion.Notify(TerminationCriterionBase::TYPE_TEST_ACCURACY, 0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());

    //first countdown
    criterion.Notify(TerminationCriterionBase::TYPE_TEST_ACCURACY, 0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //second countdown
    criterion.Notify(TerminationCriterionBase::TYPE_TEST_ACCURACY, 0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //reset
    criterion.Notify(TerminationCriterionBase::TYPE_TEST_ACCURACY, 0.6);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //first countdown
    criterion.Notify(TerminationCriterionBase::TYPE_TEST_ACCURACY, 0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //second countdown
    criterion.Notify(TerminationCriterionBase::TYPE_TEST_ACCURACY, 0.5);
    EXPECT_FALSE(criterion.IsCriterionMet());
    
    //third countdown
    criterion.Notify(TerminationCriterionBase::TYPE_TEST_ACCURACY, 0.5);
    
    EXPECT_TRUE(criterion.IsCriterionMet());
  }

}  // namespace caffe
