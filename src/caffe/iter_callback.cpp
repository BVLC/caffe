// Copyright 2014 BVLC and contributors.

#include "caffe/common.hpp"
#include "caffe/iter_callback.hpp"

// Provides indication to the solver after each iteration  if it should save
// a snapshot, continue or stop, what the learning rate should be and cetera.

namespace caffe {

// Constructor.
template<typename Dtype>
IterActions<Dtype>::IterActions():
        create_snapshot_( false ),
        continue_( false ),
        test_( false ),
        display_( false ),
        learn_rate_( 0.0 ),
        momentum_( 0.0 ),
        resume_( false ),
        resume_file_(),
        weight_decay_( 0.0 )
{
}

template<typename Dtype>
IterActions<Dtype> IterActions<Dtype>::SetResumeFile(
        const std::string& resume_file ) const
{
    IterActions actions = *this;
    actions.resume_file_ = resume_file;
    actions.resume_ = true;
    return actions;
}

template<typename Dtype>
IterActions<Dtype> IterActions<Dtype>::SetShouldSnapshot() const
{
    IterActions actions = *this;
    actions.create_snapshot_ = true;
    return actions;
}

template<typename Dtype>
IterActions<Dtype> IterActions<Dtype>::SetShouldTest() const
{
    IterActions actions = *this;
    actions.test_ = true;
    return actions;
}

template<typename Dtype>
IterActions<Dtype> IterActions<Dtype>::SetLearnRate( Dtype learn_rate ) const
{
    IterActions actions = *this;
    actions.learn_rate_ = learn_rate;
    return actions;
}

template<typename Dtype>
IterActions<Dtype> IterActions<Dtype>::SetMomentum( Dtype momentum ) const
{
    IterActions actions = *this;
    actions.momentum_ = momentum;
    return actions;
}

template<typename Dtype>
IterActions<Dtype> IterActions<Dtype>::SetWeightDecay( double weight_decay ) const
{
    IterActions actions = *this;
    actions.weight_decay_ = weight_decay;
    return actions;
}

template<typename Dtype>
IterActions<Dtype> IterActions<Dtype>::SetShouldContinue() const
{
    IterActions actions = *this;
    actions.continue_ = true;
    return actions;
}

template<typename Dtype>
IterActions<Dtype> IterActions<Dtype>::SetShouldStop() const
{
    IterActions actions = *this;
    actions.continue_ = false;
    return actions;
}

template<typename Dtype>
IterActions<Dtype> IterActions<Dtype>::SetShouldDisplay() const
{
    IterActions actions = *this;
    actions.display_ = true;
    return actions;
}

// Returns true iff a snapshot should be created.
template<typename Dtype>
bool IterActions<Dtype>::ShouldSnapshot() const
{
    return create_snapshot_;
}

template<typename Dtype>
bool IterActions<Dtype>::ShouldDisplay() const
{
    return display_;
}

template<typename Dtype>
bool IterActions<Dtype>::ShouldTest() const
{
    return test_;
}

// Returns true iff the solver should continue solving, or false to stop.
template<typename Dtype>
bool IterActions<Dtype>::ShouldContinue() const
{
    return continue_;
}

template<typename Dtype>
bool IterActions<Dtype>::ShouldResume() const
{
    return resume_;
}

template<typename Dtype>
std::string IterActions<Dtype>::GetResumeFile() const
{
    return resume_file_;
}

template<typename Dtype>
Dtype IterActions<Dtype>::GetLearningRate() const
{
    return learn_rate_;
}

template<typename Dtype>
Dtype IterActions<Dtype>::GetMomentum() const
{
    return momentum_;
}

template<typename Dtype>
Dtype IterActions<Dtype>::GetWeightDecay() const
{
    return weight_decay_;
}

//==============================================================================
// TestResult class
//==============================================================================
template<typename Dtype>
TestResult<Dtype> TestResult<Dtype>::SetLoss( Dtype loss ) const
{
  TestResult<Dtype> result = *this;
  result.loss_ = loss;
  return result;
}

// One score for each test iteration.
template<typename Dtype>
TestResult<Dtype> TestResult<Dtype>::SetScores(
                                const std::vector<Dtype>& scores ) const
{
  TestResult<Dtype> result = *this;
  result.scores_ = scores;
  return result;
}

template<typename Dtype>
boost::optional<Dtype> TestResult<Dtype>::GetLoss() const
{
  return loss_;
}

template<typename Dtype>
std::vector<Dtype> TestResult<Dtype>::GetScores() const
{
  return scores_;
}


//==============================================================================
// Current training statistics.
//==============================================================================
template<typename Dtype>
TrainingStats<Dtype> TrainingStats<Dtype>::SetIter( int iter ) const
{
    TrainingStats stats = *this;
    stats.iter_ = iter;
    return stats;
}

template<typename Dtype>
TrainingStats<Dtype> TrainingStats<Dtype>::SetLoss( Dtype loss ) const
{
    TrainingStats stats = *this;
    stats.loss_ = loss;
    return stats;
}

template<typename Dtype>
TrainingStats<Dtype> TrainingStats<Dtype>::AddTestNetResult(
                                    const TestResult<Dtype>& result ) const
{
    TrainingStats stats = *this;
    stats.testnet_results_.push_back( result );
    return stats;
}

template<typename Dtype>
int TrainingStats<Dtype>::GetIter() const
{
    return iter_;
}

template<typename Dtype>
Dtype TrainingStats<Dtype>::GetLoss() const
{
  return loss_;
}

// One TestResult instance for each test net.
template<typename Dtype>
std::vector< TestResult<Dtype> > TrainingStats<Dtype>::GetTestNetResults() const
{
  return testnet_results_;
}

INSTANTIATE_CLASS(IterActions);
INSTANTIATE_CLASS(TestResult);
INSTANTIATE_CLASS(TrainingStats);

} // namespace caffe
