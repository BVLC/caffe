// Copyright 2014 BVLC and contributors.
#pragma once

#include <boost/function.hpp>
#include <boost/optional.hpp>
#include <string>
#include <vector>

namespace caffe {

// Provides indication to the solver after each iteration  if it should save
// a snapshot, continue or stop, what the learning rate should be and cetera.
template< typename Dtype>
struct IterActions {
    // Constructor.
    IterActions();
    void SetResumeFile(const std::string& resume_file);
    void SetShouldSnapshot();
    void SetShouldTest();
    void SetLearnRate(Dtype learn_rate);
    void SetMomentum(Dtype momentum);
    void SetWeightDecay(double weight_decay);
    void SetShouldContinue();
    void SetShouldStop();
    void SetShouldDisplay();
    // True if the solver should create a snapshot.
    bool ShouldSnapshot() const;
    // True if solver should write to the log.
    bool ShouldDisplay() const;
    // True if the solver should test.
    bool ShouldTest() const;
    // True if the solver should continue running.
    bool ShouldContinue() const;
    // True if the solver should resume from the resume file.
    bool ShouldResume() const;
    // Path to the file to load when resuming.
    std::string GetResumeFile() const;
    Dtype GetLearningRate() const;
    Dtype GetMomentum() const;
    Dtype GetWeightDecay() const;

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
struct TestResult{
    void SetLoss(Dtype loss);
    // One score for each test iteration.
    void SetScores(const std::vector<Dtype>& scores);
    boost::optional<Dtype> GetLoss() const;
    std::vector<Dtype> GetScores() const;

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
struct TrainingStats {
    void SetStartIter(int start_iter);
    void SetIter(int iter);
    void SetLoss(Dtype loss);
    void AddTestNetResult(const TestResult<Dtype>& result);
    int GetIter() const;
    int GetStartIter() const;
    Dtype GetLoss() const;
    // One TestResult instance for each test net.
    std::vector< TestResult<Dtype> > GetTestNetResults() const;

 private:
    // Number of iterations when training started. Greater than zero
    // if we started by resuming from a previously-saved state.
    int start_iter_;
    // Number of iterations completed.
    int iter_;
    // Loss on the net being trained.
    Dtype loss_;
    // Results of the test net testing.
    std::vector< TestResult<Dtype> > testnet_results_;
};

// Type of the training iteration callback functor that is provided to the
// solver. The function is called by the solver to deliver
// statistics to the solver's client, and the client returns IterActions
// indicating whether training continue, if a snapshot should be saved, learning
// rate and cetera.
template <typename Dtype>
struct IterCallback {
  // Type for a function that takes a TrainingStats object, and returns
  // an IterActions object.
  typedef boost::function< IterActions<Dtype>(
                              const TrainingStats<Dtype>&)> Type;
};

}  // namespace caffe
