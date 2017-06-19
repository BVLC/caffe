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

#ifndef CAFFE_MLSL_HPP_
#define CAFFE_MLSL_HPP_

#ifdef USE_MLSL

#include <mlsl.hpp>
#include "caffe/common.hpp"

namespace caffe {
  namespace mn {

#define MLSL_DEFAULT_COLOR -1

    inline void free(void *addr) {
      return MLSL::Environment::GetEnv().Free(addr);
    }

    inline void* alloc(size_t size, int alignment) {
      return MLSL::Environment::GetEnv().Alloc(size, alignment);
    }

    inline int get_node_id() {
      return MLSL::Environment::GetEnv().GetProcessIdx();
    }

    inline int get_nodes_count() {
      return MLSL::Environment::GetEnv().GetProcessCount();
    }

    inline int get_group_id(int data_parts, int model_parts) {
      int node_id = get_node_id();
      int num_nodes = get_nodes_count();
      return (node_id % (num_nodes / data_parts)) / model_parts;
    }

    inline bool is_multinode() {
      static bool multinode{ get_nodes_count() > 1 };
      return multinode;
    }

    inline bool is_root() {
      return mn::get_node_id() == 0;
    }

    namespace detail {
      template <typename Dtype>
      inline MLSL::DataType dtype();

      template <>
      inline MLSL::DataType dtype<long double>() {
        return MLSL::DataType::DT_DOUBLE;
      }
      template <>
      inline MLSL::DataType dtype<float>() {
        return MLSL::DataType::DT_FLOAT;
      }
      template <>
      inline MLSL::DataType dtype<double>() {
        return MLSL::DataType::DT_DOUBLE;
      }
    }

    class Distribution {
    public:
      Distribution() = delete;
      Distribution & operator = (const Distribution &) = delete;
      Distribution(const Distribution &) = delete;

      Distribution(int dataParts, int modelParts, int dataColor = MLSL_DEFAULT_COLOR, int modelColor = MLSL_DEFAULT_COLOR,
                   int dataColorMax = MLSL_DEFAULT_COLOR, int modelColorMax = MLSL_DEFAULT_COLOR) :
        data_parts_(dataParts), model_parts_(modelParts),
        data_color_(dataColor), model_color_(modelColor),
        data_color_max_(dataColorMax), model_color_max_(modelColorMax) {
        if (dataColor == MLSL_DEFAULT_COLOR || modelColor == MLSL_DEFAULT_COLOR) {
          distrib_ = MLSL::Environment::GetEnv().CreateDistribution(dataParts, modelParts);
        } else {
          distrib_ = MLSL::Environment::GetEnv().CreateDistributionWithColors(dataColor, modelColor);
        }
      }
      ~Distribution() {
        MLSL::Environment::GetEnv().DeleteDistribution(distrib_);
      }
      operator MLSL::Distribution * () {
        return distrib_;
      }
      template <typename Dtype, MLSL::ReductionType Rtype, MLSL::GroupType Gtype>
      void reduce(Dtype *sendBuffer, Dtype *recvBuffer, size_t count, size_t rootIdx = 0) {
        if (skip_comm(Gtype)) return;
        MLSL::CommReq *rqts = distrib_->Reduce((void *)sendBuffer, (void*)recvBuffer, count, detail::dtype<Dtype>(), Rtype, rootIdx, Gtype);
        MLSL::Environment::GetEnv().Wait(rqts);
      }
      template <typename Dtype, MLSL::GroupType Gtype>
      void bcast(Dtype *buffer, size_t count, int rootId = 0) {
        if (skip_comm(Gtype)) return;
        MLSL::CommReq *rqts = distrib_->Bcast((void *)buffer, count, detail::dtype<Dtype>(), rootId, Gtype);
        MLSL::Environment::GetEnv().Wait(rqts);
      }
      template <typename Dtype, MLSL::ReductionType Rtype, MLSL::GroupType Gtype>
      void allreduce(Dtype *sendBuffer, Dtype *recvBuffer, size_t count) {
        if (skip_comm(Gtype)) return;
        MLSL::CommReq *rqts = distrib_->AllReduce((void *)sendBuffer, (void *)recvBuffer, count, detail::dtype<Dtype>(), Rtype, Gtype);
        MLSL::Environment::GetEnv().Wait(rqts);
      }
      template <typename Dtype, MLSL::ReductionType Rtype, MLSL::GroupType Gtype>
      void allreduce(Dtype *buffer, size_t count) {
        if (skip_comm(Gtype)) return;
        MLSL::CommReq *rqts = distrib_->AllReduce((void *)buffer, (void *)buffer, count, detail::dtype<Dtype>(), Rtype, Gtype);
        MLSL::Environment::GetEnv().Wait(rqts);
      }
      template <typename Dtype, MLSL::GroupType Gtype>
      void gather(const Dtype *sendBuffer, size_t count, Dtype *recvBuffer, size_t rootIdx = 0) {
        if (skip_comm(Gtype)) return;
        MLSL::CommReq *rqts = distrib_->Gather((void *)sendBuffer, count, (void *)recvBuffer, detail::dtype<Dtype>(), rootIdx, Gtype);
        MLSL::Environment::GetEnv().Wait(rqts);
      }
      template <typename Dtype, MLSL::GroupType Gtype>
      void scatter(Dtype *sendBuffer, Dtype *recvBuffer, size_t count, size_t rootIdx = 0) {
        if (skip_comm(Gtype)) return;
        MLSL::CommReq *rqts = distrib_->Scatter((void *)sendBuffer, (void *)recvBuffer, count, detail::dtype<Dtype>(), rootIdx, Gtype);
        MLSL::Environment::GetEnv().Wait(rqts);
      }
      template <typename Dtype, MLSL::ReductionType Rtype, MLSL::GroupType Gtype>
      void reducescatter(Dtype *sendBuffer, Dtype *recvBuffer, size_t count) {
        if (skip_comm(Gtype)) return;
        MLSL::CommReq *rqts = distrib_->ReduceScatter(sendBuffer, recvBuffer, count, detail::dtype<Dtype>(), Rtype, Gtype);
        MLSL::Environment::GetEnv().Wait(rqts);
      }
      template <typename Dtype, MLSL::GroupType Gtype>
      void allgather(Dtype *sendBuffer, size_t count, Dtype *recvBuffer) {
        if (skip_comm(Gtype)) return;
        // TODO: support allgather from MLSL
        gather<Dtype,Gtype>(sendBuffer, count, recvBuffer);
        size_t bcast_count = count;
        switch (Gtype) {
        case MLSL::GT_MODEL:
          bcast_count *= model_parts_;
          break;
        case MLSL::GT_DATA:
          bcast_count *= data_parts_;
          break;
        case MLSL::GT_GLOBAL:
          bcast_count *= model_parts_ * data_parts_;
          break;
        default:
          NOT_IMPLEMENTED;
        }
        bcast<Dtype,Gtype>(recvBuffer, bcast_count);
      }
      template <MLSL::GroupType Gtype>
      void barrier() {
        if (skip_comm(Gtype)) return;
        distrib_->Barrier(Gtype);
      }
      inline int get_data_parts() {
        return data_parts_;
      }
      inline int get_model_parts() {
        return model_parts_;
      }
      inline int get_group_id() {
        return mn::get_group_id(data_parts_, model_parts_);
      }
    private:
      inline bool skip_comm(MLSL::GroupType Gtype) {
        if (Gtype == MLSL::GT_DATA && data_color_max_ != MLSL_DEFAULT_COLOR) {
          return data_color_ > data_color_max_;
        } else if (Gtype == MLSL::GT_MODEL && model_color_max_ != MLSL_DEFAULT_COLOR) {
          return model_color_ > model_color_max_;
        } else return get_group_id() > 0;
      }

      MLSL::Distribution *distrib_{ nullptr };
      int data_parts_;
      int model_parts_;
      int data_color_;
      int model_color_;
      int data_color_max_;
      int model_color_max_;
    };

    inline void GetCanonicalMnParam(int &num_nodes, int &model_parts) {
      if (num_nodes == 0) num_nodes = mn::get_nodes_count();
      if (model_parts == 0 || model_parts > num_nodes) model_parts = num_nodes;
    }

    shared_ptr<Distribution> create_distrib(
      int dataParts, int modelParts, int dataColor = MLSL_DEFAULT_COLOR, int modelColor = MLSL_DEFAULT_COLOR,
      int dataColorMax = MLSL_DEFAULT_COLOR, int modelColorMax = MLSL_DEFAULT_COLOR);
    Distribution * get_distrib(int dataParts, int modelParts);
    Distribution * get_distrib();

    template <typename Dtype, MLSL::ReductionType Rtype = MLSL::RT_SUM>
    inline void allreduce(Dtype *sendBuffer, Dtype *recvBuffer, size_t count) {
      get_distrib()->allreduce<Dtype, Rtype, MLSL::GT_GLOBAL>(sendBuffer, recvBuffer, count);
    }
    template <typename Dtype, MLSL::ReductionType Rtype = MLSL::RT_SUM>
    inline void allreduce(Dtype *buffer, size_t count) {
      get_distrib()->allreduce<Dtype, Rtype, MLSL::GT_GLOBAL>(buffer, count);
    }
    template <typename Dtype, MLSL::ReductionType Rtype = MLSL::RT_SUM>
    inline void reduce(Dtype *buffer, size_t count, int rootId = 0) {
      get_distrib()->reduce<Dtype, Rtype, MLSL::GT_GLOBAL>(buffer, count, rootId);
    }
    template <typename Dtype>
    void bcast(Dtype *buffer, size_t count, int rootId = 0) {
      get_distrib()->bcast<Dtype, MLSL::GT_GLOBAL>(buffer, count, rootId);
    }
    template <typename Dtype>
    inline void gather(const Dtype *sendBuffer, size_t count, Dtype *recvBuffer, int rootId = 0) {
      get_distrib()->gather<Dtype, MLSL::GT_GLOBAL>(sendBuffer, count, recvBuffer, rootId);
    }
    template <typename Dtype>
    inline void scatter(Dtype *sendBuffer, Dtype *recvBuffer, size_t count, int rootId = 0) {
      get_distrib()->scatter<Dtype, MLSL::GT_GLOBAL>(sendBuffer, recvBuffer, count, rootId);
    }

    /* */
    class Session {
    public:
      Session(MLSL::PhaseType phaseType)
        : session_{ MLSL::Environment::GetEnv().CreateSession(phaseType) } {
      }
      ~Session() {
        session_->RemoveOperations();
        MLSL::Environment::GetEnv().DeleteSession(session_);
      }
      operator MLSL::Session * () {
        return session_;
      }
      void commit() {
        session_->Commit();
      }
      void set_global_minibatch_size(int global_minibatch_size) {
        session_->SetGlobalMinibatchSize(global_minibatch_size);
      }
      int get_global_minibatch_size() {
        return session_->GetGlobalMinibatchSize();
      }
      MLSL::Operation * add_operation(MLSL::OperationRegInfo *opRegInfo, MLSL::Distribution *distrib = nullptr) {
        return session_->GetOperation(session_->AddOperation(opRegInfo, distrib));
      }
      void delete_operation_reg_info(MLSL::OperationRegInfo *opRegInfo) {
        session_->DeleteOperationRegInfo(opRegInfo);
      }
      MLSL::OperationRegInfo * create_operation_reg_info(MLSL::OpType opType) {
        return session_->CreateOperationRegInfo(opType);
      }
      size_t get_operation_count() {
          return session_->GetOperationCount();
      }
      const char* get_operation_name(size_t idx) {
          return session_->GetOperation(idx)->GetName();
      }
      MLSL::Statistics * get_stats() {
          return session_->GetStats();
      }
    private:
      MLSL::Session *session_{ nullptr };
    };

    namespace train {

      inline Session & get_session() {
        static Session session{ MLSL::PT_TRAIN };
        return session;
      }
      
      inline MLSL::Operation * add_operation(MLSL::OperationRegInfo* opRegInfo, MLSL::Distribution* distrib = *get_distrib()) {
        return get_session().add_operation(opRegInfo, distrib);
      }

      inline int get_global_minibatch_size() {
        return get_session().get_global_minibatch_size();
      }

      inline void set_global_minibatch_size(int global_minibatch_size) {
        get_session().set_global_minibatch_size(global_minibatch_size);
      }

      inline void commit() {
        get_session().commit();
      }

      namespace stats {
        inline void stop() {
          get_session().get_stats()->Stop();
        }
        inline void print() {
          get_session().get_stats()->Print();
        }
        inline void reset() {
          get_session().get_stats()->Reset();
        }
        inline void start() {
          get_session().get_stats()->Start();
        }
        inline bool is_started() {
          return get_session().get_stats()->IsStarted();
        }
        inline unsigned long long get_isolation_comm_time(size_t idx) {
          return get_session().get_stats()->GetIsolationCommCycles(idx);
        }
        inline size_t get_comm_size(size_t idx) {
          return get_session().get_stats()->GetCommSize(idx);
        }
        inline unsigned long long get_comm_time(size_t idx) {
          return get_session().get_stats()->GetCommCycles(idx);
        }
        inline unsigned long long get_compute_time(size_t idx) {
          return get_session().get_stats()->GetComputeCycles(idx);
        }
        inline unsigned long long get_total_isolation_comm_time() {
          return get_session().get_stats()->GetTotalIsolationCommCycles();
        }
        inline size_t get_total_comm_size() {
          return get_session().get_stats()->GetTotalCommSize();
        }
        inline unsigned long long get_total_comm_time() {
          return get_session().get_stats()->GetTotalCommCycles();
        }
        inline unsigned long long get_total_compute_time() {
          return get_session().get_stats()->GetTotalComputeCycles();
        }

      }
    }

    class OpRegInfo {
    public:
      OpRegInfo() = delete;
      OpRegInfo & operator = (const OpRegInfo &) = delete;
      OpRegInfo(const OpRegInfo &) = delete;

      OpRegInfo(OpRegInfo &&) = default;
      OpRegInfo & operator = (OpRegInfo &&) = default;

      explicit OpRegInfo(Session& session, MLSL::OpType opType)
        : opRegInfo_{ session.create_operation_reg_info(opType) },
          session_(session) {
      }
      ~OpRegInfo() {
        session_.delete_operation_reg_info(opRegInfo_);
      }
      operator MLSL::OperationRegInfo * () {
        return opRegInfo_;
      }
      void set_name(std::string name) {
        opRegInfo_->SetName(name.c_str());
      }
      template <typename Dtype>
      void add_input(int featureMapCount, int featureMapSize) {
        opRegInfo_->AddInput(featureMapCount, featureMapSize, detail::dtype<Dtype>());
      }
      template <typename Dtype>
      void add_output(int featureMapCount, int featureMapSize) {
        opRegInfo_->AddOutput(featureMapCount, featureMapSize, detail::dtype<Dtype>());
      }
      template <typename Dtype>
      void add_parameter_set(int kernelCount, int kernelSize,
        bool distributedUpdate = false)
      {
        opRegInfo_->AddParameterSet(kernelCount, kernelSize, detail::dtype<Dtype>(),
           distributedUpdate);
      }
    private:
      MLSL::OperationRegInfo *opRegInfo_{ nullptr };
      Session &session_;
    };

  }  // namespace mn
}  // namespace caffe

#endif // USE_MLSL

#endif   // CAFFE_MLSL_HPP_
