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

#include <boost/make_shared.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <algorithm>
#include <utility>
#include <vector>
#include "caffe/internode/configuration.hpp"
#include "caffe/internode/tree_cluster.hpp"
#include "caffe/multinode/BlobInfo.hpp"

namespace caffe {

namespace {

using internode::RemoteId;
using internode::Waypoint;

struct BlobConstInfoImpl : BlobConstInfo {
  typedef vector<int> LayerSizes;
  typedef vector<LayerSizes> NetSizes;

  const NetSizes sizes;
  const vector<int> layers_parts;
  const int total_parts;

  BlobConstInfoImpl(const NetSizes& sizes, const vector<int>& parts, int total)
    : sizes(sizes)
    , layers_parts(parts)
    , total_parts(total) {
  }

  virtual uint32_t parts(int layer_id, int blob_id) const {
    CHECK_GE(blob_id, 0);
    CHECK(blob_id < blobs(layer_id));
    return sizes[layer_id][blob_id];
  }

  virtual uint32_t blobs(int layer_id) const {
    CHECK_GE(layer_id, 0);
    CHECK(layer_id < layers());
    return sizes[layer_id].size();
  }

  virtual uint32_t parts(int layer_id) const {
    CHECK_GE(layer_id, 0);
    CHECK(layer_id < layers());
    return layers_parts[layer_id];
  }

  virtual uint32_t parts() const {
    return total_parts;
  }

  virtual uint32_t layers() const {
    return sizes.size();
  }
  virtual bool needs_syncing(int layer_id) const {
    CHECK_GE(layer_id, 0);
    CHECK(layer_id < layers());
    return blobs(layer_id) > 0;
  }
};

template <typename Dtype>
class BlobAccessorImpl : public BlobAccessor<Dtype> {
  const shared_ptr<Solver<Dtype> > solver;

 public:
  explicit BlobAccessorImpl(shared_ptr<Solver<Dtype> > solver)
    : solver(solver) {}

  virtual Blob<Dtype>* get_blob(int layer_id, int blob_id) {
    CHECK_GE(layer_id, 0);
    CHECK_GE(blob_id, 0);
    CHECK(layer_id < solver->net()->layers().size());
    CHECK(blob_id < solver->net()->layers()[layer_id]->blobs().size());
    return solver->net()->layers()[layer_id]->blobs()[blob_id].get();
  }
};


struct BlobSyncInfoImpl : BlobSyncInfo {
  typedef boost::unordered_map<RemoteId, int> RemoteInfoMap;
  typedef boost::unordered_set<RemoteId> RemotesSet;
  const shared_ptr<BlobConstInfo> const_info;
  RemoteInfoMap remotes;
  vector<BlobSyncInfo::Handler*> handlers;
  mutable boost::mutex mtx;

  typedef std::pair<RemoteId, int> PartKey;
  typedef boost::unordered_set<PartKey> ReceivedParts;

  struct PartInfo {
    RemotesSet received_from;
    uint32_t version;
  };
  vector<vector<vector<PartInfo> > > parts;

  std::vector<uint32_t> current_versions;
  std::vector<ReceivedParts> received_for_layer;
  int max_parts;

  explicit BlobSyncInfoImpl(shared_ptr<BlobConstInfo> const_info)
    : const_info(const_info)
    , current_versions(const_info->layers(), 0u)
    , received_for_layer(const_info->layers())
    , max_parts(calculate_max_parts()) {
    parts.resize(const_info->layers());
    for (int i = 0; i < const_info->layers(); ++i) {
      parts[i].resize(const_info->blobs(i));
      for (int j = 0; j < const_info->blobs(i); ++j) {
        parts[i][j].resize(const_info->parts(i, j));
      }
    }
  }

  uint32_t calculate_max_parts() const {
    uint32_t ret = 0;
    for (int i = 0; i < const_info->layers(); ++i) {
      for (int j = 0; j < const_info->blobs(i); ++j) {
        ret = std::max(ret, const_info->parts(i, j));
      }
    }
    return ret;
  }

  int get_id(int blob_id, int part) const {
    return blob_id * max_parts + part;
  }

  virtual bool received(RemoteId from,
                        int layer_id,
                        int blob_id,
                        int part_id,
                        uint32_t version) {
    PartKey key = PartKey(from, get_id(blob_id, part_id));
    add_remote(from);
    PartInfo& part_info = parts[layer_id][blob_id][part_id];
    {
      boost::mutex::scoped_lock lock(mtx);
      DLOG(INFO) << "BlobInfo received from " << from << ": "
        << layer_id << " " << blob_id << " " << part_id
        << " of version " << version
        << " current: " << part_info.version;
      if (version < part_info.version) return false;
      if (version > part_info.version) {
        part_info.received_from.clear();
        part_info.version = version;
      }
      if (version > current_versions[layer_id]) {
        current_versions[layer_id] = version;
        received_for_layer[layer_id].clear();
      }
      if (!received_for_layer[layer_id].insert(key).second)
        return false;
      CHECK(part_info.received_from.insert(from).second);
      DLOG(INFO) << "BlobInfo::received for layer " << layer_id
        << ": " << received_for_layer[layer_id].size()
        << "/" << (const_info->parts(layer_id) * remotes.size())
        << " [remotes: " << remotes.size() << "]";
    }

    if (synced_check_and_callback(layer_id, blob_id, part_id)
        && synced_check_and_callback(layer_id)) {
      synced_check_and_callback();
    }
    return true;
  }

  virtual void register_synced_handler(Handler* handler) {
    boost::mutex::scoped_lock lock(mtx);
    handlers.push_back(handler);
  }

  virtual uint32_t received_version(
      internode::RemoteId from, int layer_id, int blob_id, int part_id) const {
    boost::mutex::scoped_lock lock(mtx);
    const PartInfo& part_info = parts[layer_id][blob_id][part_id];
    if (part_info.received_from.count(from) > 0)
      return part_info.version;
    if (part_info.version == 0) return 0;
    return part_info.version - 1;
  }


  virtual void add_remote(RemoteId id) {
    {
      boost::mutex::scoped_lock lock(mtx);
      if (remotes.insert(std::make_pair(id, 0)).second) {
        VLOG(2) << "added remote " << id
          << " there are now "
          << remotes.size() << " remotes";
      } else {
        return;
      }
    }
    synced_check_and_callback();
  }

  virtual void remove_remote(RemoteId id) {
    {
      boost::mutex::scoped_lock lock(mtx);
      remotes.erase(id);
      for (int i = 0; i < const_info->layers(); ++i) {
        if (!const_info->needs_syncing(i)) continue;
        for (int j = 0; j < const_info->blobs(i); ++j) {
          for (int k = 0; k < const_info->parts(i, j); ++k) {
            received_for_layer[i].erase(PartKey(id, get_id(j, k)));
            parts[i][j][k].received_from.erase(id);
          }
        }
      }
    }
    for (int i = 0; i < parts.size(); ++i) {
      if (!const_info->needs_syncing(i)) continue;
      for (int j = 0; j < parts[i].size(); ++j) {
        for (int k = 0; k < parts[i][j].size(); ++k) {
          synced_check_and_callback(i, j, k);
        }
      }
      synced_check_and_callback(i);
    }
    synced_check_and_callback();
  }

 private:
  bool is_synced(int layer_id) const {
    boost::mutex::scoped_lock lock(mtx);
    return (received_for_layer[layer_id].size()
            >= const_info->parts(layer_id) * remotes.size());
  }

  bool synced_check_and_callback(int layer_id, int blob_id, int part_id) {
    PartInfo& part = parts[layer_id][blob_id][part_id];
    vector<BlobSyncInfo::Handler*> to_call;
    bool ret = false;
    {
      boost::mutex::scoped_lock lock(mtx);
      if (part.received_from.size() >= remotes.size()) {
        to_call = handlers;
        ret = true;
      }
    }

    for (int i = 0; i < to_call.size(); ++i) {
      to_call[i]->synced(layer_id, blob_id, part_id, part.version);
    }
    return ret;
  }

  bool synced_check_and_callback(int layer_id) {
    if (is_synced(layer_id)) {
      vector<BlobSyncInfo::Handler*> handlers_to_call;
      uint32_t version = 0;
      {
        boost::mutex::scoped_lock lock(mtx);
        handlers_to_call = handlers;
        version = current_versions[layer_id];
      }
      for (int i = 0; i < handlers_to_call.size(); ++i) {
        handlers_to_call[i]->synced(layer_id, version);
      }
      return true;
    }
    return false;
  }

  void synced_check_and_callback() {
    uint32_t version = UINT_MAX;
    vector<BlobSyncInfo::Handler*> handlers_to_call;
    {
      for (int i = 0; i < const_info->layers(); ++i) {
        if (!const_info->needs_syncing(i)) continue;
        if (!is_synced(i)) return;
        boost::mutex::scoped_lock lock(mtx);
        if (version == UINT_MAX) version = current_versions[i];
        else if (version != current_versions[i]) return;
        handlers_to_call = handlers;
      }
    }
    for (int i = 0; i < handlers_to_call.size(); ++i) {
      handlers_to_call[i]->synced(version);
    }
  }
};

template <typename Dtype>
shared_ptr<BlobConstInfo> create_const_info_impl(
    shared_ptr<Solver<Dtype> > solver,
    size_t elements_per_packet) {
  const int layers = solver->net()->layers().size();

  vector<vector<int> > sizes;
  vector<int> parts;
  int total_parts = 0;

  sizes.resize(layers);
  parts.resize(layers);

  for (int i = 0; i < layers; ++i) {
    if (solver->net()->get_layer_learnable_param_ids(i).empty()) continue;

    const int blobs = solver->net()->layers()[i]->blobs().size();
    sizes[i].resize(blobs);
    for (int j = 0; j < blobs; ++j) {
      const int blob_size = solver->net()->layers()[i]->blobs()[j]->count();
      const int blob_parts =
        (blob_size + elements_per_packet - 1) / elements_per_packet;
      sizes[i][j] = blob_parts;
      parts[i] += blob_parts;
      total_parts += blob_parts;
    }
  }

  return boost::make_shared<BlobConstInfoImpl>(sizes, parts, total_parts);
}

}  // namespace

template <typename Dtype>
shared_ptr<BlobConstInfo> BlobInfoFactory<Dtype>::create_const_info(
    shared_ptr<Solver<Dtype> > solver, size_t elements_per_packet) {
  return create_const_info_impl(solver, elements_per_packet);
}

template <typename Dtype>
shared_ptr<BlobSyncInfo> BlobInfoFactory<Dtype>::create_sync_info(
    shared_ptr<BlobConstInfo> const_info) {
  return boost::make_shared<BlobSyncInfoImpl>(const_info);
}

template <typename Dtype>
shared_ptr<BlobAccessor<Dtype> > BlobInfoFactory<Dtype>::create_blob_accessor(
    shared_ptr<Solver<Dtype> > solver) {
  return boost::make_shared<BlobAccessorImpl<Dtype> >(solver);
}

INSTANTIATE_CLASS(BlobInfoFactory);

}  // namespace caffe
