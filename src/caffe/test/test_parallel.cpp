#include <cstring>

#include "gtest/gtest.h"

#include "caffe/parallel.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template<typename Dtype>
class DistSyncTest: public ::testing::Test {
};

TYPED_TEST_CASE(DistSyncTest, TestDtypes);

TYPED_TEST(DistSyncTest, TestNothing) {
  // The first test case of a test suite takes the longest time
  //   due to the set up overhead.
}

//TYPED_TEST(DistSyncTest, TestMasterIndex) {
//  vector<shared_ptr<Blob<TypeParam> > > blobs();
//  blobs.push_back(new shared_ptr(new Blob<TypeParam>(1, 1, 1, 1)));
//  Params params(blobs);
//
//  vector<int> nodes();
//  nodes.push_back(0);
//  nodes.push_back(1);
//  nodes.push_back(2);
//  nodes.push_back(3);
//
//  uint32_t chunks = 1000;
//  for(int index = 0; index < nodes.size(); ++index) {
//    DistSync<TypeParam, int> sync(params, nodes, nodes, chunks);
//    sync.Init(index);
//
//    for(uint32_t chunk = 0; chunk < chunks; ++chunk)
//      EXPECT(
//          (sync.master(chunk) == index)
//          ==
//          (chunk >= sync.own_start_ && chunk < sync.own_until_));
//  }
//}

// test buffers are the same
//bool ready = true;
//for (int i = 0; i < solvers.size(); ++i)
//  if (!solvers[i])
//    ready = false;
//if (ready) {
//  for (int i = 0; i < solvers.size(); ++i) {
//    shared_ptr<Net<float> > n0 = solvers[0]->net();
//    shared_ptr<Net<float> > ni = solvers[i]->net();
//    vector<shared_ptr<Blob<float> > >& p0 = n0->params();
//    vector<shared_ptr<Blob<float> > >& pi = ni->params();
//    for (int j = 0; j < p0.size(); ++j)
//      CHECK(pi[j]->cpu_data() == p0[j]->cpu_data());
//  }
//  shared_ptr<Net<float> > n0 = solvers_[0]->net();
//  vector<shared_ptr<Blob<float> > >& p0 = n0->params();
//  for (int j = 0; j < p0.size(); ++j)
//    for (int k = 0; k < p0[j]->count(); ++k)
//      CHECK(!isnan(p0[j]->cpu_data()[k])) << " NAN";
//}


}  // namespace caffe
