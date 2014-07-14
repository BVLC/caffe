// Copyright 2014 BVLC and contributors.

#include <string>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/iter_callback.hpp"

// Provides indication to the solver after each iteration  if it should save
// a snapshot, continue or stop, what the learning rate should be and cetera.

namespace caffe {

// Constructor.
template<typename Dtype>
IterActions<Dtype>::IterActions():
        create_snapshot_(false),
        continue_(false),
        test_(false),
        display_(false),
        learn_rate_(0.0),
        momentum_(0.0),
        resume_(false),
        resume_file_(),
        weight_decay_(0.0) {}

template<typename Dtype>
void IterActions<Dtype>::SetResumeFile(
        const std::string& resume_file ) {
  resume_file_ = resume_file;
  resume_ = true;
}

template<typename Dtype>
void IterActions<Dtype>::SetShouldSnapshot() {
  create_snapshot_ = true;
}

template<typename Dtype>
void IterActions<Dtype>::SetShouldTest() {
  test_ = true;
}

template<typename Dtype>
void IterActions<Dtype>::SetLearnRate(Dtype learn_rate) {
  learn_rate_ = learn_rate;
}

template<typename Dtype>
void IterActions<Dtype>::SetMomentum(Dtype momentum) {
  momentum_ = momentum;
}

template<typename Dtype>
void IterActions<Dtype>::SetWeightDecay(double weight_decay) {
  weight_decay_ = weight_decay;
}

template<typename Dtype>
void IterActions<Dtype>::SetShouldContinue() {
  continue_ = true;
}

template<typename Dtype>
void IterActions<Dtype>::SetShouldStop() {
  continue_ = false;
}

template<typename Dtype>
void IterActions<Dtype>::SetShouldDisplay() {
  display_ = true;
}

template<typename Dtype>
bool IterActions<Dtype>::ShouldSnapshot() const {
  return create_snapshot_;
}

template<typename Dtype>
bool IterActions<Dtype>::ShouldDisplay() const {
  return display_;
}

template<typename Dtype>
bool IterActions<Dtype>::ShouldTest() const {
  return test_;
}

template<typename Dtype>
bool IterActions<Dtype>::ShouldContinue() const {
  return continue_;
}

template<typename Dtype>
bool IterActions<Dtype>::ShouldResume() const {
  return resume_;
}

template<typename Dtype>
std::string IterActions<Dtype>::GetResumeFile() const {
  return resume_file_;
}

template<typename Dtype>
Dtype IterActions<Dtype>::GetLearningRate() const {
  return learn_rate_;
}

template<typename Dtype>
Dtype IterActions<Dtype>::GetMomentum() const {
  return momentum_;
}

template<typename Dtype>
Dtype IterActions<Dtype>::GetWeightDecay() const {
  return weight_decay_;
}

//==============================================================================
// TestResult class
//==============================================================================
template<typename Dtype>
void TestResult<Dtype>::SetLoss(Dtype loss) {
  loss_ = loss;
}

template<typename Dtype>
void TestResult<Dtype>::SetScores(const std::vector<Dtype>& scores) {
  scores_ = scores;
}

template<typename Dtype>
boost::optional<Dtype> TestResult<Dtype>::GetLoss() const {
  return loss_;
}

template<typename Dtype>
std::vector<Dtype> TestResult<Dtype>::GetScores() const {
  return scores_;
}

template<typename Dtype>
void TrainingStats<Dtype>::SetIter(int iter) {
  iter_ = iter;
}

template<typename Dtype>
void TrainingStats<Dtype>::SetLoss(Dtype loss) {
  loss_ = loss;
}

template<typename Dtype>
void TrainingStats<Dtype>::AddTestNetResult(
                                    const TestResult<Dtype>& result) {
  testnet_results_.push_back(result);
}

template<typename Dtype>
int TrainingStats<Dtype>::GetIter() const {
  return iter_;
}

template<typename Dtype>
Dtype TrainingStats<Dtype>::GetLoss() const {
  return loss_;
}

template<typename Dtype>
std::vector< TestResult<Dtype> >
TrainingStats<Dtype>::GetTestNetResults() const {
  return testnet_results_;
}

INSTANTIATE_CLASS(IterActions);
INSTANTIATE_CLASS(TestResult);
INSTANTIATE_CLASS(TrainingStats);

}  // namespace caffe
