#ifdef USE_MPI
#include <mpi.h>
#include "caffe/internode/mpiutil.hpp"
#endif

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/optional.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <cmath>
#include <utility>
#include <vector>

#include "caffe/internode/broadcast_callback.hpp"
#include "caffe/internode/configuration.hpp"
#include "caffe/internode/tree_cluster.hpp"
#include "caffe/serialization/BlobCodec.hpp"


const int MSG_TAG = 1972;

namespace caffe {
namespace internode {

extern boost::asio::io_service& get_io_service(boost::shared_ptr<Daemon>);

#ifdef USE_MPI

typedef boost::function<void(bool, int, int)> RequestCallback;
typedef std::pair<MPI_Request, RequestCallback> MpiRequestWithCallback;

class MpiTreeClient : public TreeWaypoint {
  boost::shared_ptr<Daemon> daemon;
  std::vector<Handler*> handlers;
  std::vector<MpiRequestWithCallback> requests;
  std::vector<char> buffer;
  boost::recursive_mutex mtx;

  void set_recv() {
    boost::recursive_mutex::scoped_lock lock(mtx);
    requests.push_back(std::make_pair(
      MPI_Request(), boost::bind(&MpiTreeClient::received, this, _1, _2, _3)));
    MPI_Irecv(
            &buffer.front(), buffer.size(),
            MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
            &requests.back().first);
    DLOG(INFO) << "**** (set_recv) requests: " << requests.size();
  }

  void received(bool ok, int size, int sender) {
    if (ok) {
      boost::recursive_mutex::scoped_lock lock(mtx);
      DLOG(INFO) << "[proc " << id() << "] received buffer of size: " << size
        << ", checksum: " << (uint16_t)check_sum<uint8_t>(
          reinterpret_cast<uint8_t*>(&buffer.front()), size);
      if (sender == parent()) {
        for (int i = 0; i < handlers.size(); ++i) {
          handlers[i]->received_from_parent(&buffer.front(), size);
        }
      } else {
        for (int i = 0; i < handlers.size(); ++i) {
          handlers[i]->received_from_child(&buffer.front(), size, sender);
        }
      }
    } else {
      LOG(ERROR) << "RECEIVED FAILED";
    }

    set_recv();
  }

 public:
  explicit MpiTreeClient(boost::shared_ptr<Daemon> daemon)
      : daemon(daemon) {
    post(daemon);
  }

  virtual boost::shared_ptr<Daemon> get_daemon() {
    return daemon;
  }

  virtual void set_buffer_size(size_t max_packet_size) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    buffer.resize(max_packet_size);
    set_recv();
  }

  virtual void async_send_to_parent(const char* buffer,
                                    size_t size,
                                    SentCallback callback) {
    RemoteId parent_id = parent();

    boost::recursive_mutex::scoped_lock lock(mtx);
    requests.push_back(std::make_pair(
      MPI_Request(), boost::bind(callback, _1)));
    MPI_Isend(const_cast<char*>(buffer),
              size,
              MPI_CHAR,
              parent_id,
              MSG_TAG,
              MPI_COMM_WORLD,
              &requests.back().first);
    DLOG(INFO) << "**** (async_send_to_parent) requests: " << requests.size();
  }

  virtual void async_send_to_children(const char* buff,
                                      size_t size,
                                      SentCallback callback) {
    std::vector<RemoteId> children_ids = children();

    boost::recursive_mutex::scoped_lock lock(mtx);
    DLOG(INFO) << "[proc " << id() << "] sending buffer of size: " << size
      << ", checksum: " << (uint16_t)check_sum<uint8_t>(
        reinterpret_cast<const uint8_t*>(buff), size);


    BroadcastCallback<SentCallback> broadcast_callback(callback);
    for (int i = 0; i < children_ids.size(); ++i) {
      requests.push_back(std::make_pair(MPI_Request(), broadcast_callback));
      MPI_Isend(const_cast<char*>(buff),
                size,
                MPI_CHAR,
                children_ids[i],
                MSG_TAG,
                MPI_COMM_WORLD,
                &requests.back().first);
    }
    DLOG(INFO) << "**** (async_send_to_children) requests: " << requests.size();
  }

  virtual void register_receive_handler(Handler* handler) {
    handlers.push_back(handler);
  }

  virtual RemoteId id() const {
    return mpi_get_current_proc_rank();
  }

  virtual std::vector<RemoteId> children() const {
    std::vector<RemoteId> children;
    RemoteId parent = id();
    int count = mpi_get_comm_size();

    if (count < 2) return children;

    if (parent * 2 + 1 < count) children.push_back( parent*2+1 );
    if (parent * 2 + 2 < count) children.push_back( parent*2+2 );

    return children;
  }

  virtual RemoteId parent() const {
    RemoteId current = id();

    if (current == 0) return 0;

    return floor( (current-1)/2 );
  }

  virtual void poll_one(shared_ptr<Daemon> daemon) {
    boost::optional<MpiRequestWithCallback> request;
    bool op_result = false;
    RemoteId sender = 0;
    int size = 0;
    {
      boost::recursive_mutex::scoped_lock lock(mtx);
      for (int i = 0; i < requests.size(); ++i) {
        MPI_Status status;
        int flag = 0, result = MPI_Test(&requests[i].first, &flag, &status);
        if (flag) {
          request = requests[i];
          requests.erase(requests.begin() + i);

          sender = status.MPI_SOURCE;
          result = MPI_Get_count(&status, MPI_CHAR, &size);
          if (result == MPI_SUCCESS) {
            DLOG(INFO) << "**** (poll_one) requests: " << requests.size();
            op_result = true;
            break;
          } else {
            LOG(ERROR) << "ERROR: " << mpi_get_error_string(result);
          }
        }
      }
    }
    if (request)
      request->second(op_result, size, sender);

    post(daemon);
  }

  virtual void post(shared_ptr<Daemon> daemon) {
    get_io_service(daemon).post(
      boost::bind(&MpiTreeClient::poll_one, this, daemon));
  }
};

TreeWaypoint* TreeWaypoint::get_instance() {
  static boost::shared_ptr<Daemon> daemon = create_communication_daemon();
  static MpiTreeClient instance(daemon);
  return &instance;
}

#else

TreeWaypoint* TreeWaypoint::get_instance() {
  LOG(ERROR) << "can't use MPI";
  throw std::runtime_error("can't use MPI");
}

#endif

}  // namespace internode
}  // namespace caffe

