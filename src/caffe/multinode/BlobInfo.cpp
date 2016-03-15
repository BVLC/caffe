#include <boost/make_shared.hpp>
#include <boost/unordered_map.hpp>
#include <algorithm>
#include <utility>
#include <vector>
#include "caffe/internode/configuration.hpp"
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

struct RemoteSyncInfo {
  int iter_size;
  vector<vector<vector<uint32_t> > > versions;
  shared_ptr<BlobConstInfo> const_info;
  uint32_t attached_at;

  RemoteSyncInfo(shared_ptr<BlobConstInfo> const_info,
                 uint32_t attached_at)
    : iter_size(1)
    , const_info(const_info)
    , attached_at(attached_at) {
    versions.resize(const_info->layers());
    for (int i = 0; i < const_info->layers(); ++i) {
      versions[i].resize(const_info->blobs(i));
      for (int j = 0; j < const_info->blobs(i); ++j) {
        versions[i][j].resize(const_info->parts(i, j), 0);
      }
    }
  }

  uint32_t min_version(int layer_id) {
    uint32_t ret = UINT_MAX;
    CHECK_GE(layer_id, 0);
    CHECK(layer_id < const_info->layers());
    for (int j = 0; j < const_info->blobs(layer_id); ++j) {
      for (int k = 0; k < const_info->parts(layer_id, j); ++k) {
        ret = std::min(ret, versions[layer_id][j][k]);
      }
    }
    return ret;
  }

  uint32_t max_version() {
    uint32_t ret = 0;
    for (int i = 0; i < const_info->layers(); ++i) {
      for (int j = 0; j < const_info->blobs(i); ++j) {
        for (int k = 0; k < const_info->parts(i, j); ++k) {
          ret = std::max(ret, versions[i][j][k]);
        }
      }
    }
    return ret;
  }

  std::pair<uint32_t, bool> synced(int layer_id) {
    uint32_t min_version = UINT_MAX;
    uint32_t max_version = 0;
    CHECK_GE(layer_id, 0);
    CHECK(layer_id < const_info->layers());
    for (int j = 0; j < const_info->blobs(layer_id); ++j) {
      for (int k = 0; k < const_info->parts(layer_id, j); ++k) {
        min_version = std::min(min_version, versions[layer_id][j][k]);
        max_version = std::max(max_version, versions[layer_id][j][k]);
      }
    }
    return std::make_pair(min_version, min_version == max_version);
  }

  std::pair<uint32_t, bool> synced() {
    uint32_t min_version = UINT_MAX;
    uint32_t max_version = 0;
    for (int i = 0; i < const_info->layers(); ++i) {
      for (int j = 0; j < const_info->blobs(i); ++j) {
        for (int k = 0; k < const_info->parts(i, j); ++k) {
          min_version = std::min(min_version, versions[i][j][k]);
          max_version = std::max(max_version, versions[i][j][k]);
        }
      }
    }
    return std::make_pair(min_version, min_version == max_version);
  }
};

struct BlobSyncInfoImpl : BlobSyncInfo {
  typedef boost::unordered_map<RemoteId, shared_ptr<RemoteSyncInfo>
                              > RemoteInfoMap;
  const shared_ptr<BlobConstInfo> const_info;
  RemoteInfoMap remotes;
  vector<BlobSyncInfo::Handler*> handlers;

  explicit BlobSyncInfoImpl(shared_ptr<BlobConstInfo> const_info)
    : const_info(const_info) {
  }

  virtual bool received(RemoteId from,
                        int layer_id,
                        int blob_id,
                        int part,
                        uint32_t version,
                        int iters) {
    add_remote(from);
    remotes[from]->iter_size = iters;

    DLOG(INFO) << "received update from " << from
      << "{" << layer_id << ", " << blob_id << ", " << part << "} "
      << " of version: " << version
      << " and previous is: "
      << remotes[from]->versions[layer_id][blob_id][part];

    uint32_t prev_version = remotes[from]->versions[layer_id][blob_id][part];
    if (prev_version >= version) return false;
    remotes[from]->versions[layer_id][blob_id][part] = version;
    if (synced_check_and_callback(layer_id))
      synced_check_and_callback();
    return true;
  }

  virtual bool synced_check_and_callback(int layer_id) {
    typedef RemoteInfoMap::const_iterator It;
    if (remotes.empty()) return true;
    int count = 0;
    uint32_t version = remotes.begin()->second->synced(layer_id).first;
    for (It it = remotes.begin(); it != remotes.end(); ++it) {
      std::pair<uint32_t, bool> synced = it->second->synced(layer_id);
      if (synced.first < it->second->attached_at) continue;
      count++;
      if (!synced.second) return false;
      if (synced.first != version) return false;
    }
    if (count == 0) return true;
    for (int i = 0; i < handlers.size(); ++i) {
      handlers[i]->synced(layer_id, version);
    }
    return true;
  }

  virtual void synced_check_and_callback() {
    typedef RemoteInfoMap::const_iterator It;
    if (remotes.empty()) return;
    uint32_t version = remotes.begin()->second->synced().first;
    int count = 0;
    for (It it = remotes.begin(); it != remotes.end(); ++it) {
      std::pair<uint32_t, bool> synced = it->second->synced();
      if (synced.first < it->second->attached_at) continue;
      ++count;
      if (!synced.second) return;
      if (synced.first != version) return;
    }
    if (count == 0) return;
    for (int i = 0; i < handlers.size(); ++i) {
      handlers[i]->synced(version);
    }
  }

  virtual void register_synced_handler(Handler* handler) {
    handlers.push_back(handler);
  }

  virtual uint32_t min_received_version(int layer_id) const {
    typedef RemoteInfoMap::const_iterator It;
    uint32_t ret = UINT_MAX;
    for (It it = remotes.begin(); it != remotes.end(); ++it) {
      ret = std::min(ret, it->second->min_version(layer_id));
    }
    return ret;
  }

  virtual uint32_t received_version(
    internode::RemoteId from, int layer_id, int blob_id, int part) const {
    CHECK_GE(part, 0);
    CHECK_LT(part, const_info->parts(layer_id, blob_id));
    typedef RemoteInfoMap::const_iterator It;
    It it = remotes.find(from);
    if (it == remotes.end()) return 0u;
    return it->second->versions[layer_id][blob_id][part];
  }

  virtual uint32_t max_version() const {
    typedef RemoteInfoMap::const_iterator It;
    uint32_t ret = 0;
    for (It it = remotes.begin(); it != remotes.end(); ++it) {
      ret = std::max(ret, it->second->max_version());
    }
    return ret;
  }

  virtual int get_total_iters() const {
    typedef RemoteInfoMap::const_iterator It;
    int ret = 0;
    for (It it = remotes.begin(); it != remotes.end(); ++it) {
      ret += it->second->iter_size;
    }
    return ret;
  }


  virtual void add_remote(RemoteId id) {
    if (remotes.count(id) == 0) {
      remotes[id].reset(new RemoteSyncInfo(
        const_info, max_version()));
      VLOG(1) << "added remote at version " << max_version()
        << " there are now " << remotes.size() << " remotes";
    }
  }

  virtual void remove_remote(RemoteId id) {
    remotes.erase(id);
    for (int i = 0; i < const_info->layers(); ++i) {
      if (!const_info->needs_syncing(i)) continue;
      synced_check_and_callback(i);
    }
    synced_check_and_callback();
  }
};

template <typename Dtype>
shared_ptr<BlobConstInfo> create_const_info(shared_ptr<Solver<Dtype> > solver,
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
BlobInfo<Dtype>::BlobInfo(shared_ptr<Solver<Dtype> > solver,
                          size_t elements_per_packet)
  : const_info(create_const_info(solver, elements_per_packet))
  , sync_info(boost::make_shared<BlobSyncInfoImpl>(const_info)) {
}

template <typename Dtype>
shared_ptr<BlobConstInfo> BlobInfo<Dtype>::get_const_info() const {
  return const_info;
}

template <typename Dtype>
shared_ptr<BlobSyncInfo> BlobInfo<Dtype>::get_sync_info() const {
  return sync_info;
}

INSTANTIATE_CLASS(BlobInfo);

}  // namespace caffe

