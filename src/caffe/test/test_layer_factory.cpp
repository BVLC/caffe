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

#include <map>
#include <string>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class LayerFactoryTest : public MultiDeviceTest<TypeParam> {};

TYPED_TEST_CASE(LayerFactoryTest, TestDtypesAndDevices);

TYPED_TEST(LayerFactoryTest, TestCreateLayer) {
  typedef typename TypeParam::Dtype Dtype;
  typename LayerRegistry<Dtype>::CreatorRegistry& registry =
      LayerRegistry<Dtype>::Registry();
  shared_ptr<Layer<Dtype> > layer;
  for (typename LayerRegistry<Dtype>::CreatorRegistry::iterator iter =
       registry.begin(); iter != registry.end(); ++iter) {
    // Special case: PythonLayer is checked by pytest
    if (iter->first == "Python") { continue; }
    LayerParameter layer_param;
    // Data layers expect a DB
    if (iter->first == "Data" || iter->first == "AnnotatedData" || iter->first == "BoxData") {
#ifdef USE_LEVELDB
      string tmp;
      MakeTempDir(&tmp);
      boost::scoped_ptr<db::DB> db(db::GetDB(DataParameter_DB_LEVELDB));
      db->Open(tmp, db::NEW);
      db->Close();
      layer_param.mutable_data_param()->set_source(tmp);
#else
      continue;
#endif  // USE_LEVELDB
    }
    layer_param.set_type(iter->first);
    layer = LayerRegistry<Dtype>::CreateLayer(layer_param);
    EXPECT_EQ(iter->first, layer->type());
  }
}

}  // namespace caffe
