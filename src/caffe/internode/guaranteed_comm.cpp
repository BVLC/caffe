#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/icl/interval_set.hpp>
#include <boost/make_shared.hpp>
#include <boost/optional.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>
#include <boost/variant.hpp>
#include <boost/weak_ptr.hpp>
#include <algorithm>
#include <deque>
#include <string>
#include <utility>
#include <vector>
#include "caffe/internal_thread.hpp"
#include "caffe/internode/guaranteed_comm.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
namespace internode {

extern boost::asio::io_service& get_io_service(boost::shared_ptr<Daemon>);

namespace {

const int TIME_TO_RESEND_IN_UMS = 1;
const int BUFFER_SIZE = 65535;
const int MAX_BUFFERS = 4000;
const size_t MAX_CHECKSUM_SIZE = 150;
const size_t MAX_UNACKED = 100;

class CommBuffers {
  const size_t buffer_size;
  mutable boost::mutex mtx;

  std::vector<char> total_buffer;
  std::deque<uint32_t> available_buffers;
  std::vector<char*> all_buffers;
  std::vector<bool> available;

  std::pair<char*, uint32_t> pop_impl() {
    boost::mutex::scoped_lock lock(mtx);
    CHECK(!available_buffers.empty());
    uint32_t id = available_buffers.front();
    available_buffers.pop_front();
    available[id] = false;
    return std::make_pair(all_buffers[id], id);
  }

  void give_back(uint32_t id) {
    boost::mutex::scoped_lock lock(mtx);
    CHECK(id < all_buffers.size());
    available_buffers.push_back(id);
    available[id] = true;
  }

 public:
  CommBuffers(size_t buffer_size, size_t max_buffers)
    : buffer_size(buffer_size)
    , total_buffer(max_buffers * buffer_size, '\0')
    , all_buffers(max_buffers, NULL)
    , available(max_buffers, true) {
    for (int i = 0; i < max_buffers; ++i) {
      all_buffers[i] = &total_buffer.front() + i * buffer_size;
      available_buffers.push_back(i);
    }
  }

  class Buffer {
    CommBuffers* holder;
    std::pair<char*, uint32_t> repr;
   public:
    explicit Buffer(CommBuffers* holder)
      : holder(holder)
      , repr(holder->pop_impl()) {
    }

    ~Buffer() {
      holder->give_back(repr.second);
    }

    char* ptr() {
      return repr.first;
    }
  };

  boost::shared_ptr<Buffer> pop() {
    CHECK(!fully_utilized());
    return boost::shared_ptr<Buffer>(new Buffer(this));
  }

  bool fully_utilized() const {
    boost::mutex::scoped_lock lock(mtx);
    return available_buffers.empty();
  }
};

typedef uint64_t UID;
UID generate_uid() {
  static boost::mutex mtx;
  static UID gen_uid = 1;
  boost::mutex::scoped_lock guard(mtx);
  return gen_uid++;
}

class PacketCodec {
  const char msg_indicator;
  const char ack_indicator;
  const size_t ack_buffer_size;
  const size_t msg_header_size;

 public:
  typedef unsigned char Checksum;

  PacketCodec()
    : msg_indicator('M')
    , ack_indicator('A')
    , ack_buffer_size(sizeof(UID) + sizeof(msg_indicator))
    , msg_header_size(ack_buffer_size + sizeof(Checksum)) {
    CHECK(sizeof(msg_indicator) == sizeof(ack_indicator));
  }

  size_t encode_ack(char* dest, UID id) const {
    dest[0] = ack_indicator;
    *reinterpret_cast<UID*>(dest + 1) = id;
    return ack_buffer_size;
  }

  size_t encode_msg(
    char* dest_buffer, const char* buffer, size_t size, UID id) const {
    encode_ack(dest_buffer, id);
    caffe_copy<char>(size, buffer, dest_buffer + msg_header_size);
    Checksum checksum = buffer[0];
    for (int i = 0; i < std::min(MAX_CHECKSUM_SIZE, size); ++i) {
      checksum = checksum ^ (unsigned char)buffer[i];
    }
    dest_buffer[0] = msg_indicator;
    dest_buffer[ack_buffer_size] = static_cast<char>(checksum);
    return msg_header_size + size;
  }

  std::pair<UID, bool> decode_ack(
      char* buffer, size_t size) const {
    CHECK(size >= ack_buffer_size);
    CHECK((buffer[0] == ack_indicator) || (buffer[0] == msg_indicator));
    return std::make_pair(*reinterpret_cast<UID*>(buffer + 1),
                          (buffer[0] == ack_indicator));
  }

  std::pair<size_t, char*> decode_msg(char* buffer, size_t size) const {
    CHECK(size >= msg_header_size);
    CHECK((buffer[0] == ack_indicator) || (buffer[0] == msg_indicator));
    Checksum checksum = buffer[msg_header_size];
    Checksum received_checksum = (Checksum)buffer[ack_buffer_size];
    size_t to_check = std::min(msg_header_size + MAX_CHECKSUM_SIZE, size);
    for (int i = msg_header_size; i < to_check; ++i) {
      checksum = checksum ^ (Checksum)buffer[i];
    }
    if (checksum != received_checksum) {
      LOG(ERROR) << "checksum incorrect";
    }
    return ((buffer[0] == msg_indicator) && (checksum == received_checksum)) ?
      std::make_pair(size - msg_header_size, buffer + msg_header_size) :
      std::make_pair(0lu, static_cast<char*>(NULL));
  }

  size_t ack_size() const {
    return ack_buffer_size;
  }
};

class Resender {
  boost::shared_ptr<Waypoint> waypoint;

  struct SentBuffer {
    UID uid;
    char* ptr;
    size_t size;
    boost::posix_time::ptime sent_on;
    boost::weak_ptr<CommBuffers::Buffer> weak;
  };
  std::deque<SentBuffer> sent_buffers;
  typedef boost::unordered_map<
    UID, boost::shared_ptr<CommBuffers::Buffer> > Pushed;
  Pushed all_pushed;

  void send(SentBuffer buffer) {
    waypoint->async_send(
      buffer.ptr, buffer.size, boost::bind(&Resender::resent, this));
    buffer.sent_on = boost::posix_time::second_clock::local_time();
    sent_buffers.push_back(buffer);
    DLOG(INFO) << "resending msg " << buffer.uid;
  }

  void resent() {
    resend_some();
  }

 public:
  Resender(boost::shared_ptr<Daemon> daemon,
           boost::shared_ptr<Waypoint> waypoint)
    : waypoint(waypoint) {
  }

  void push(boost::shared_ptr<CommBuffers::Buffer> buffer,
            size_t size,
            UID uid) {
    boost::weak_ptr<CommBuffers::Buffer> weak(all_pushed[uid] = buffer);
    SentBuffer sent = {
      uid, buffer->ptr(), size,
      boost::posix_time::second_clock::local_time(), weak};
    sent_buffers.push_back(sent);
  }

  void resend_some() {
    while (!sent_buffers.empty()) {
      SentBuffer oldest = sent_buffers.front();
      if (oldest.weak.expired()) {
        sent_buffers.pop_front();
        continue;
      }
      boost::posix_time::time_duration diff =
        boost::posix_time::second_clock::local_time() - oldest.sent_on;
      if (diff.total_microseconds() < TIME_TO_RESEND_IN_UMS)
        break;
      sent_buffers.pop_front();
      send(oldest);
    }
  }

  void ack(UID msg_id) {
    all_pushed.erase(msg_id);

    DLOG(INFO) << "received ack for: " << msg_id
               << " from " << waypoint->address()
               << "(" << waypoint->id() << ") "
               << " left: " << all_pushed.size();
  }

  size_t size() const {
    return all_pushed.size();
  }
};

struct SentItem {
  Waypoint::SentCallback callback;
  bool result;
};

struct SendItem {
  RemoteId id;
  boost::shared_ptr<CommBuffers::Buffer> buffer;
  size_t size;
  UID uid;
};

struct RecvItem {
  RemoteId id;
  boost::shared_ptr<CommBuffers::Buffer> buffer;
  size_t size;
};

struct DisconnectedItem {
  RemoteId id;
};

struct AcceptedItem {
  RemoteId id;
  std::string address;
};

typedef boost::variant<
  RecvItem, SentItem, SendItem, DisconnectedItem, AcceptedItem> Item;

Item make_sent_item(Waypoint::SentCallback callback, bool result) {
  SentItem ret = {callback, result};
  return ret;
}

Item make_send_item(RemoteId id,
                    boost::shared_ptr<CommBuffers::Buffer> buffer,
                    size_t size,
                    UID uid) {
  SendItem ret = {id, buffer, size, uid};
  return ret;
}

Item make_accept_item(RemoteId id, std::string address) {
  AcceptedItem ret = {id, address};
  return ret;
}

Item make_disconnected_item(RemoteId id) {
  DisconnectedItem ret = {id};
  return ret;
}

class Queue {
  boost::mutex mtx;
  std::deque<Item> queue;
  CommBuffers buffs;

 public:
  explicit Queue(size_t buffer_size) : buffs(BUFFER_SIZE, MAX_BUFFERS) {
  }

  void push(Item item) {
    boost::mutex::scoped_lock lock(mtx);
    queue.push_back(item);
  }

  void push_front(Item item) {
    boost::mutex::scoped_lock lock(mtx);
    queue.push_front(item);
  }

  boost::optional<Item> pop() {
    boost::mutex::scoped_lock lock(mtx);
    if (queue.empty()) return boost::none;
    Item first = queue.front();
    queue.pop_front();
    return first;
  }

  Item make_recv_item(RemoteId id, char* buffer, size_t size) {
    boost::shared_ptr<CommBuffers::Buffer> own_buffer = buffs.pop();
    caffe_copy<char>(size, buffer, own_buffer->ptr());
    RecvItem ret = {id, own_buffer, size};
    return ret;
  }
};

class GuaranteedWaypoint : public InternalThread, public Waypoint::Handler {
  typedef Waypoint::SentCallback SentCallback;

  PacketCodec codec;
  boost::shared_ptr<Queue> send_queue;
  boost::shared_ptr<Queue> recv_queue;
  boost::shared_ptr<Daemon> daemon;
  boost::shared_ptr<Waypoint> waypoint;
  CommBuffers ack_buffers;
  Resender resender_;
  std::vector<Waypoint::Handler*> handlers;

  boost::icl::interval_set<UID> received_msgs;

  bool sending;

  void send_ack(UID msg_id) {
    boost::shared_ptr<CommBuffers::Buffer> ack_buffer = ack_buffers.pop();
    size_t size = codec.encode_ack(ack_buffer->ptr(), msg_id);
    DLOG(INFO) << "sending ack for: " << msg_id;
    waypoint->async_send(
      ack_buffer->ptr(), size,
      boost::bind(&GuaranteedWaypoint::sent_ack, this, _1, ack_buffer));
  }

  void sent_ack(bool, boost::shared_ptr<CommBuffers::Buffer>) {
  }

  void sent(bool ok, boost::shared_ptr<CommBuffers::Buffer>) {
    sending = false;
  }

  void send_msg() {
    if (sending) return;
    if (resender_.size() > MAX_UNACKED) return;
    boost::optional<Item> next = send_queue->pop();
    if (!next) return;
    const SendItem& to_send = boost::get<SendItem>(*next);
    CHECK(to_send.id == waypoint->id());
    async_send(to_send.buffer, to_send.size, to_send.uid);
  }

 protected:
  virtual void received(char* buffer, size_t size, Waypoint* from) {
    std::pair<UID, bool> ack = codec.decode_ack(buffer, size);
    if (ack.second) {
      resender_.ack(ack.first);
    } else {
      DLOG(INFO) << "received msg: " << ack.first;
      std::pair<size_t, char*> msg = codec.decode_msg(buffer, size);
      if (msg.second == NULL) return;
      send_ack(ack.first);
      if (!already_received(ack.first)) {
        recv_queue->push(
          recv_queue->make_recv_item(from->id(), msg.second, msg.first));
        mark_as_received(ack.first);
      }
    }
    resender_.resend_some();
  }

 public:
  GuaranteedWaypoint(boost::shared_ptr<Queue> send_queue,
                     boost::shared_ptr<Queue> recv_queue,
                     boost::shared_ptr<Daemon> daemon,
                     boost::shared_ptr<Waypoint> non_guaranteed_waypoint,
                     bool set_timer)
    : send_queue(send_queue)
    , recv_queue(recv_queue)
    , daemon(daemon)
    , waypoint(non_guaranteed_waypoint)
    , ack_buffers(codec.ack_size(), MAX_BUFFERS)
    , resender_(daemon, non_guaranteed_waypoint)
    , sending(false) {
    CHECK(!non_guaranteed_waypoint->guaranteed_comm());
    if (set_timer) {
      create_timer(
        daemon,
        TIME_TO_RESEND_IN_UMS,
        boost::bind(&GuaranteedWaypoint::tick, this),
        true);
    }
  }

  virtual void async_send(boost::shared_ptr<CommBuffers::Buffer> buffer,
                          size_t size,
                          UID uid) {
    sending = true;
    waypoint->async_send(
      buffer->ptr(), size,
      boost::bind(&GuaranteedWaypoint::sent, this, _1, buffer));
    resender_.push(buffer, size, uid);
  }

  bool already_received(UID uid) {
    return boost::icl::contains(received_msgs, uid);
  }
  void mark_as_received(UID uid) {
    received_msgs.insert(uid);
  }

  virtual void tick() {
    resender_.resend_some();
  }

  virtual void register_receive_handler(Waypoint::Handler* handler) {
    handlers.push_back(handler);
  }

  virtual size_t max_packet_size() const {
    return waypoint->max_packet_size() - codec.ack_size();
  }

  Waypoint& raw() {
    return *waypoint;
  }

  Resender& resender() {
    return resender_;
  }

  virtual void InternalThreadEntry() {
    while (true) {
      poll_one(daemon);
      resender_.resend_some();
      send_msg();
    }
  }
};

class GuaranteedMultiWaypoint : public InternalThread
                              , public Waypoint::Handler
                              , public MultiWaypoint::Handler {
  typedef Waypoint::SentCallback SentCallback;

  PacketCodec codec;
  boost::shared_ptr<Queue> send_queue;
  boost::shared_ptr<Queue> recv_queue;
  boost::shared_ptr<Daemon> daemon;
  boost::shared_ptr<Waypoint> waypoint;
  CommBuffers ack_buffers;
  typedef boost::tuple<const char*, size_t, SentCallback> ItemToSend;
  std::deque<ItemToSend> to_send;
  bool sending;

  typedef boost::unordered_map<RemoteId, boost::shared_ptr<GuaranteedWaypoint>
    > Clients;
  Clients clients;

  vector<Waypoint::Handler*> receive_handlers;
  vector<MultiWaypoint::Handler*> accept_handlers;

  void send_ack(RemoteId to, UID msg_id) {
    boost::shared_ptr<CommBuffers::Buffer> ack_buffer = ack_buffers.pop();
    DLOG(INFO) << "sending ack for: " << msg_id;
    size_t size = codec.encode_ack(ack_buffer->ptr(), msg_id);
    clients[to]->raw().async_send(
      ack_buffer->ptr(), size,
      boost::bind(&GuaranteedMultiWaypoint::sent_ack, this, _1, ack_buffer));
  }

  void sent_ack(bool, boost::shared_ptr<CommBuffers::Buffer>) {
  }

  bool not_valid_client(RemoteId id) {
    return clients.find(id) == clients.end();
  }

  void sent(bool ok, boost::shared_ptr<CommBuffers::Buffer>) {
    sending = false;
  }

  void send_msg() {
    if (sending) return;
    if (clients.empty()) return;
    if (clients.begin()->second->resender().size() > MAX_UNACKED) return;
    boost::optional<Item> next = send_queue->pop();
    if (!next) return;
    const SendItem& to_send = boost::get<SendItem>(*next);
    if (to_send.id != waypoint->id()) {
      Clients::iterator it = clients.find(to_send.id);
      if (it == clients.end()) return;
      clients[to_send.id]->async_send(
        to_send.buffer, to_send.size, to_send.uid);
      return;
    }

    sending = true;
    waypoint->async_send(to_send.buffer->ptr(), to_send.size,
      boost::bind(&GuaranteedMultiWaypoint::sent, this, _1, to_send.buffer));
    typedef Clients::iterator It;
    for (It it = clients.begin(); it != clients.end(); ++it) {
      it->second->resender().push(to_send.buffer, to_send.size, to_send.uid);
    }
  }

 protected:
  virtual void received(char* buffer, size_t size, Waypoint* from) {
    std::pair<UID, bool> ack = codec.decode_ack(buffer, size);
    typedef Clients::iterator It;
    It client = clients.find(from->id());
    if (client == clients.end()) return;
    if (ack.second) {
      client->second->resender().ack(ack.first);
    } else {
      std::pair<size_t, char*> msg = codec.decode_msg(buffer, size);
      if (msg.second == NULL) return;
      send_ack(from->id(), ack.first);
      if (!client->second->already_received(ack.first)) {
        recv_queue->push(
          recv_queue->make_recv_item(client->first, msg.second, msg.first));
        client->second->mark_as_received(ack.first);
      }
    }
    client->second->resender().resend_some();
    send_msg();
  }

  virtual void accepted(boost::shared_ptr<Waypoint> client) {
    clients[client->id()] =
      boost::make_shared<GuaranteedWaypoint>(
        send_queue, recv_queue, daemon, client, false);
    recv_queue->push(make_accept_item(client->id(), client->address()));
  }

  virtual void disconnected(RemoteId id) {
    clients.erase(id);
    recv_queue->push(make_disconnected_item(id));
  }

  void tick() {
    typedef Clients::iterator It;
    for (It it = clients.begin(); it != clients.end(); ++it) {
      it->second->resender().resend_some();
    }
  }

 public:
  GuaranteedMultiWaypoint(boost::shared_ptr<Queue> send_queue,
                          boost::shared_ptr<Queue> recv_queue,
                          boost::shared_ptr<Daemon> daemon,
                          boost::shared_ptr<MultiWaypoint> waypoint)
    : send_queue(send_queue)
    , recv_queue(recv_queue)
    , daemon(daemon)
    , waypoint(waypoint)
    , ack_buffers(codec.ack_size(), MAX_BUFFERS)
    , sending(false) {
    create_timer(
      daemon,
      TIME_TO_RESEND_IN_UMS,
      boost::bind(&GuaranteedMultiWaypoint::tick, this),
      true);
  }

  virtual size_t max_packet_size() const {
    return waypoint->max_packet_size() - codec.ack_size();
  }

  virtual void InternalThreadEntry() {
    while (true) {
      poll_one(daemon);
      send_msg();
    }
  }
};

class ExternalClientWaypoint : public Waypoint, public boost::static_visitor<> {
  PacketCodec codec;
  boost::shared_ptr<Queue> send_queue;
  boost::shared_ptr<Queue> recv_queue;
  const RemoteId id_;
  const string address_;
  const size_t max_packet_size_;
  std::vector<Waypoint::Handler*> receive_handlers;
  boost::shared_ptr<CommBuffers> msg_buffers;

 public:
  void operator()(RecvItem recv) {
    received(recv.buffer->ptr(), recv.size);
  }
  void operator()(SentItem sent) {
    sent.callback(sent.result);
  }
  template <typename T>
  void operator()(T) {
  }

  ExternalClientWaypoint(boost::shared_ptr<Queue> send_queue,
                         boost::shared_ptr<Queue> recv_queue,
                         RemoteId id,
                         string address,
                         size_t max_packet_size)
    : send_queue(send_queue)
    , recv_queue(recv_queue)
    , id_(id)
    , address_(address)
    , max_packet_size_(max_packet_size)
    , msg_buffers(new CommBuffers(BUFFER_SIZE, MAX_BUFFERS)) {
  }

  void received(char* buffer, size_t size) {
    for (int i = 0; i < receive_handlers.size(); ++i)
      receive_handlers[i]->received(buffer, size, this);
  }

  virtual void async_send(const char* buffer,
                          size_t size,
                          Waypoint::SentCallback callback) {
    boost::shared_ptr<CommBuffers::Buffer> next_buffer = msg_buffers->pop();
    UID uid = generate_uid();
    size_t encoded_size =
      codec.encode_msg(next_buffer->ptr(), buffer, size, uid);
    CHECK(encoded_size <= max_packet_size());
    send_queue->push(make_send_item(id(), next_buffer, encoded_size, uid));
    recv_queue->push_front(make_sent_item(callback, true));
  }

  virtual void register_receive_handler(Waypoint::Handler* handler) {
    receive_handlers.push_back(handler);
  }

  virtual RemoteId id() const {
    return id_;
  }
  virtual string address() const {
    return address_;
  }
  virtual bool guaranteed_comm() const {
    return true;
  }
  virtual size_t max_packet_size() const {
    return max_packet_size_;
  }

  void poll_one(boost::shared_ptr<Daemon> daemon) {
    boost::optional<Item> next = recv_queue->pop();
    if (!next) return post(daemon);
    boost::apply_visitor(*this, *next);
    post(daemon);
  }

  void post(boost::shared_ptr<Daemon> daemon) {
    get_io_service(daemon).post(
      boost::bind(&ExternalClientWaypoint::poll_one, this, daemon));
  }
};

class ExternalMultiWaypoint : public MultiWaypoint
                            , public boost::static_visitor<> {
  typedef Waypoint::SentCallback SentCallback;

  PacketCodec codec;
  boost::shared_ptr<Queue> send_queue;
  boost::shared_ptr<Queue> recv_queue;
  const RemoteId id_;
  const string address_;
  const size_t max_packet_size_;

  boost::shared_ptr<CommBuffers> msg_buffers;
  std::vector<Waypoint::Handler*> receive_handlers;
  std::vector<MultiWaypoint::Handler*> accept_handlers;

  typedef boost::unordered_map<
    RemoteId, boost::shared_ptr<ExternalClientWaypoint> > Clients;
  Clients clients;

  virtual void received(char* buffer, size_t size, RemoteId from) {
    clients[from]->received(buffer, size);
  }

 public:
  void operator()(RecvItem item) {
    clients[item.id]->received(item.buffer->ptr(), item.size);
  }
  virtual void operator()(SentItem sent) {
    sent.callback(sent.result);
  }
  void operator()(AcceptedItem item) {
    clients[item.id] =
      boost::make_shared<ExternalClientWaypoint>(
        send_queue, recv_queue, item.id, item.address, max_packet_size());
    for (int i = 0; i < accept_handlers.size(); ++i)
      accept_handlers[i]->accepted(clients[item.id]);
    for (int i = 0; i < receive_handlers.size(); ++i)
      clients[item.id]->register_receive_handler(receive_handlers[i]);
  }
  void operator()(DisconnectedItem item) {
    clients.erase(item.id);
    for (int i = 0; i < accept_handlers.size(); ++i)
      accept_handlers[i]->disconnected(item.id);
  }
  template <typename T>
  void operator()(T) {
  }

  ExternalMultiWaypoint(boost::shared_ptr<Queue> send_queue,
                        boost::shared_ptr<Queue> recv_queue,
                        RemoteId id,
                        string address,
                        size_t max_packet_size)
    : send_queue(send_queue)
    , recv_queue(recv_queue)
    , id_(id)
    , address_(address)
    , max_packet_size_(max_packet_size)
    , msg_buffers(new CommBuffers(BUFFER_SIZE, MAX_BUFFERS)) {
  }

  virtual void async_send(const char* buffer,
                          size_t size,
                          SentCallback callback) {
    boost::shared_ptr<CommBuffers::Buffer> next_buffer = msg_buffers->pop();
    UID uid = generate_uid();
    size_t encoded_size =
      codec.encode_msg(next_buffer->ptr(), buffer, size, uid);
    CHECK(encoded_size <= max_packet_size());
    send_queue->push(make_send_item(id(), next_buffer, encoded_size, uid));
    recv_queue->push_front(make_sent_item(callback, true));
  }

  virtual void register_receive_handler(Waypoint::Handler* handler) {
    receive_handlers.push_back(handler);
    typedef Clients::iterator It;
    for (It it = clients.begin(); it != clients.end(); ++it) {
      it->second->register_receive_handler(handler);
    }
  }
  virtual void register_peer_change_handler(MultiWaypoint::Handler* handler) {
    accept_handlers.push_back(handler);
  }

  virtual RemoteId id() const {
    return id_;
  }
  virtual string address() const {
    return address_;
  }
  virtual bool guaranteed_comm() const {
    return true;
  }
  virtual size_t max_packet_size() const {
    return max_packet_size_;
  }

  void poll_one(boost::shared_ptr<Daemon> daemon) {
    boost::optional<Item> next = recv_queue->pop();
    if (!next) return post(daemon);
    boost::apply_visitor(*this, *next);
    post(daemon);
  }

  void post(boost::shared_ptr<Daemon> daemon) {
    get_io_service(daemon).post(
      boost::bind(&ExternalMultiWaypoint::poll_one, this, daemon));
  }
};

}  // namespace

boost::shared_ptr<Waypoint> configure_guaranteed_client(
    boost::shared_ptr<Daemon> external_daemon,
    boost::shared_ptr<Daemon> internal_daemon,
    boost::shared_ptr<Waypoint> client) {
  CHECK(!client->guaranteed_comm());

  boost::shared_ptr<Queue> send_queue(new Queue(client->max_packet_size()));
  boost::shared_ptr<Queue> recv_queue(new Queue(client->max_packet_size()));
  boost::shared_ptr<GuaranteedWaypoint> internal =
    boost::make_shared<GuaranteedWaypoint>(
      send_queue, recv_queue, internal_daemon, client, true);
  client->register_receive_handler(internal.get());

  ExternalClientWaypoint* ret = new ExternalClientWaypoint(
      send_queue, recv_queue,
      client->id(), client->address(), internal->max_packet_size());
  ret->post(external_daemon);
  internal->StartInternalThread();
  return boost::shared_ptr<ExternalClientWaypoint>(internal, ret);
}

boost::shared_ptr<MultiWaypoint> configure_guaranteed_server(
    boost::shared_ptr<Daemon> external_daemon,
    boost::shared_ptr<Daemon> internal_daemon,
    boost::shared_ptr<MultiWaypoint> server) {
  CHECK(!server->guaranteed_comm());

  boost::shared_ptr<Queue> send_queue(new Queue(server->max_packet_size()));
  boost::shared_ptr<Queue> recv_queue(new Queue(server->max_packet_size()));
  boost::shared_ptr<GuaranteedMultiWaypoint> internal =
    boost::make_shared<GuaranteedMultiWaypoint>(
      send_queue, recv_queue, internal_daemon, server);
  server->register_receive_handler(internal.get());
  server->register_peer_change_handler(internal.get());

  ExternalMultiWaypoint* ret = new ExternalMultiWaypoint(
      send_queue, recv_queue,
      server->id(), server->address(), internal->max_packet_size());
  ret->post(external_daemon);
  internal->StartInternalThread();
  return boost::shared_ptr<ExternalMultiWaypoint>(internal, ret);
}

}  // namespace internode
}  // namespace caffe

