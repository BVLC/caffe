/*
 * All modification made by Intel Corporation: © 2016 Intel Corporation
 *
 * All contributions by the University of California:
 * Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014, 2015, the respective contributors
 * All rights reserved.
 * For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
 *
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Intel Corporation nor the names of its contributors
 *       may be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef USE_MLSL

#include <map>
#include "boost/thread/mutex.hpp"
#include "caffe/multinode/mlsl.hpp"
#ifdef ENABLE_WEIGHT_GRAD_COMPRESSION
#include "caffe/util/compression_util.hpp"
#endif

namespace caffe {
  namespace mn {
    int nGroup = 1;
    int nServer = 0;
    boost::mutex distrib_lock;
    std::map<std::pair<int,int>, boost::shared_ptr<Distribution>> *distrib_map;

    void init(int* argc, char **argv[]) {
      static class initialize {
      public:
        initialize(int* argc, char** argv[]) {
          MLSL::Environment::GetEnv().Init(argc, argv);
          distrib_map =
            new std::map<std::pair<int,int>, boost::shared_ptr<Distribution>>();
          if (use_param_server()) {
            if (is_param_server()) {
              // this is for paramter servers
              MLSL::Environment::GetEnv().Configure("color=0");
            }
            else {
              // this is for workers
              int group_id = get_group_id();
              std::string config_str = "color=" + std::to_string(group_id + 1);
              MLSL::Environment::GetEnv().Configure(config_str.c_str());
            }
          }
#ifdef ENABLE_WEIGHT_GRAD_COMPRESSION
          get_weight_grad_compress_info();
#endif
        }
        ~initialize() {
          delete distrib_map;
          MLSL::Environment::GetEnv().Finalize();
        }
      } __init{ argc, argv };
    }
    
    template<>  
    MPI_Datatype DtypeToMPIDtype<float>() { return MPI_FLOAT; }

    template<> 
    MPI_Datatype DtypeToMPIDtype<double>() { return MPI_DOUBLE; }

    template<>  
    MLSL::DataType DtypeToMLSLDtype<float>() { return MLSL::DT_FLOAT; }

    template<> 
    MLSL::DataType DtypeToMLSLDtype<double>() { return MLSL::DT_DOUBLE; }

    boost::shared_ptr<Distribution> create_distrib(
      int dataParts, int modelParts, int dataColor, int modelColor,
      int dataColorMax, int modelColorMax) {
      return boost::shared_ptr<Distribution>(
        new Distribution(dataParts, modelParts, dataColor, modelColor,
          dataColorMax, modelColorMax));
    }

    boost::shared_ptr<Distribution> create_distrib(int dataParts, int modelParts) {
      int node_id = get_node_id();
      int num_nodes = get_group_size();
      int modelColor = node_id / modelParts;
      int dataColor = node_id % (num_nodes / dataParts);
      return create_distrib(dataParts, modelParts, dataColor, modelColor);
    }

    boost::shared_ptr<Distribution> create_distrib() {
      return create_distrib(get_group_size(), 1);
    }

    Distribution * get_distrib(int dataParts, int modelParts) {
      boost::mutex::scoped_lock l(distrib_lock);
      std::pair<int,int> key = std::make_pair(dataParts, modelParts);
      if (distrib_map->find(key) == distrib_map->end()) {
        (*distrib_map)[key] = create_distrib(dataParts, modelParts);
      }
      return (*distrib_map)[key].get();
    }

    Distribution * get_distrib() {
      return get_distrib(get_group_size(), 1);
    }
  }
}

#endif /* USE_MLSL */
