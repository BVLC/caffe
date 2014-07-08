#pragma once

#include <string>
#include <boost/function.hpp>
#include <boost/optional.hpp>

namespace caffe
{

// Provides indication to the solver after each iteration  if it should save
// a snapshot, continue, and cetera.
template< typename Dtype>
struct IterActions
{
    // Constructor.
    IterActions():
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

    IterActions SetResumeFile( const std::string& resume_file ) const
    {
        IterActions actions = *this;
        actions.resume_file_ = resume_file;
        actions.resume_ = true;
        return actions;
    }

    IterActions SetShouldSnapshot() const
    {
        IterActions actions = *this;
        actions.create_snapshot_ = true;
        return actions;
    }

    IterActions SetShouldTest() const
    {
        IterActions actions = *this;
        actions.test_ = true;
        return actions;
    }

    IterActions SetLearnRate( Dtype learn_rate ) const
    {
        IterActions actions = *this;
        actions.learn_rate_ = learn_rate;
        return actions;
    }

    IterActions SetMomentum( Dtype momentum ) const
    {
        IterActions actions = *this;
        actions.momentum_ = momentum;
        return actions;
    }

    IterActions SetWeightDecay( double weight_decay ) const
    {
        IterActions actions = *this;
        actions.weight_decay_ = weight_decay;
        return actions;
    }

    IterActions SetShouldContinue() const
    {
        IterActions actions = *this;
        actions.continue_ = true;
        return actions;
    }

    IterActions SetShouldStop() const
    {
        IterActions actions = *this;
        actions.continue_ = false;
        return actions;
    }

    IterActions SetShouldDisplay() const
    {
        IterActions actions = *this;
        actions.display_ = true;
        return actions;
    }

    // Returns true iff a snapshot should be created.
    bool ShouldSnapshot() const
    {
        return create_snapshot_;
    }

    bool ShouldDisplay() const
    {
        return display_;
    }

    bool ShouldTest() const
    {
        return test_;
    }

    // Returns true iff the solver should continue solving, or false to stop.
    bool ShouldContinue() const
    {
        return continue_;
    }

    bool ShouldResume() const
    {
        return resume_;
    }

    std::string GetResumeFile() const
    {
        return resume_file_;
    }

    Dtype GetLearningRate() const
    {
        return learn_rate_;
    }

    Dtype GetMomentum() const
    {
        return momentum_;
    }

    Dtype GetWeightDecay() const
    {
        return weight_decay_;
    }
private:
    // True iff the solver should save a snapshot.
    bool create_snapshot_;
    // True iff the solver should continue training.
    bool continue_;
    bool test_;
    bool display_;
    Dtype learn_rate_;
    Dtype momentum_;
    bool resume_;
    std::string resume_file_;
    Dtype weight_decay_;
};

template<typename Dtype>
struct TestResult
{
    TestResult SetLoss( Dtype loss ) const
    {
        TestResult result = *this;
        result.loss_ = loss;
        return result;
    }

    // One score for each test iteration.
    TestResult SetScores( const std::vector<Dtype>& scores ) const
    {
        TestResult result = *this;
        result.scores_ = scores;
        return result;
    }

    boost::optional<Dtype> GetLoss() const
    {
        return loss_;
    }

    std::vector<Dtype> GetScores() const
    {
        return scores_;
    }
private:
    // Loss on the test net, if param_.test_compute_loss_ was true.
    // Otherwise the loss is not present.
    boost::optional<Dtype> loss_;
    // Scores of the test net on each "test iteration". So size is equal
    // to param_.test_iter(test_net_id).
    std::vector<Dtype> scores_;
};

// Current training statistics.
template<typename Dtype>
struct TrainingStats
{
    TrainingStats SetIter( int iter ) const
    {
        TrainingStats stats = *this;
        stats.iter_ = iter;
        return stats;
    }

    TrainingStats SetLoss( Dtype loss ) const
    {
        TrainingStats stats = *this;
        stats.loss_ = loss;
        return stats;
    }

    TrainingStats AddTestNetResult( const TestResult<Dtype>& result ) const
    {
        TrainingStats stats = *this;
        stats.testnet_results_.push_back( result );
        return stats;
    }

    int GetIter() const
    {
        return iter_;
    }

    Dtype GetLoss() const
    {
        return loss_;
    }

    // One TestResult instance for each test net.
    std::vector< TestResult<Dtype> > GetTestNetResults() const
    {
        return testnet_results_;
    }
private:
    // Number of iterations completed.
    int iter_;
    // Loss on the net being trained.
    Dtype loss_;
    // Results of the test net testing.
    std::vector< TestResult<Dtype> > testnet_results_;
};

// Type of the training iteration callback functor.  Called by the solver to deliver
// statistics to the solver's client, and the client returns IterActions
// indicating whether training continue, if a snapshot should be saved, learning
// rate and cetera.
template <typename Dtype>
struct IterCallback
{
    typedef boost::function< IterActions<Dtype>( const TrainingStats<Dtype>& )> Type;
};

//template class IterCallback<float>;
//template class IterCallback<double>;

//template class IterActions<float>;
//template class IterActions<double>;

//template class TrainingStats<float>;
//template class TrainingStats<double>;

//template class TestResult<float>;
//template class TestResult<double>;

} // namespace caffe
