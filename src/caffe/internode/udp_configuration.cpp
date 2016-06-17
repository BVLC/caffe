#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/unordered_map.hpp>
#include <algorithm>
#include <deque>
#include <string>
#include <vector>
#include "caffe/internode/broadcast_callback.hpp"
#include "caffe/internode/communication.hpp"
#include "caffe/internode/configuration.hpp"
#include "caffe/internode/guaranteed_comm.hpp"

#define udp_tick_time_us 50e+4
#define udp_disconnect_time_us 50e+5
#define udp_max_datagram_size 65507

namespace caffe {
namespace internode {

extern boost::asio::io_service& get_io_service(boost::shared_ptr<Daemon>);

struct MsgType {
  static const char Connect;
  static const char Heartbeat;
  static const char Message;
  static const char Shutdown;
};
const char MsgType::Connect = 'C';
const char MsgType::Heartbeat = 'H';
const char MsgType::Message = 'M';
const char MsgType::Shutdown = 'S';

namespace {

template <typename Endpoint>
std::string get_address(const Endpoint& ep) {
  try {
    std::string ip = ep.address().to_string();
    unsigned short port = ep.port();

    return "udp://" + ip + ":" + boost::lexical_cast<string>(port);
  } catch(...) {
    return "udp://...";
  }
}

void do_nothing(bool = false) {
}

struct SendQueueItem {
  Waypoint::SentCallback callback;
  boost::array<boost::asio::const_buffer, 2> bufs;
  boost::asio::ip::udp::endpoint endpoint;

  SendQueueItem(const char* buffer,
                size_t size,
                Waypoint::SentCallback callback,
                boost::asio::ip::udp::endpoint endpoint)
    : callback(callback)
    , endpoint(endpoint) {
    bufs[0] = boost::asio::buffer(&MsgType::Message, 1);
    bufs[1] = boost::asio::buffer(buffer, size);
  }

  SendQueueItem(const char& type,
                boost::asio::ip::udp::endpoint endpoint)
    : callback(&do_nothing)
    , endpoint(endpoint) {
    bufs[0] = boost::asio::buffer(&type, sizeof(type));
    bufs[1] = boost::asio::buffer(&type, 0);
  }
};

template <bool RaiseErrorOnDisconnect>
class SendQueue
  : public boost::enable_shared_from_this<SendQueue<RaiseErrorOnDisconnect> > {
  typedef boost::enable_shared_from_this<SendQueue<RaiseErrorOnDisconnect> >
          EnabledShared;

  std::deque<SendQueueItem> queue;
  boost::shared_ptr<boost::asio::ip::udp::socket> socket;
  bool sending;

  void send() {
    if (sending) return;
    if (queue.empty()) return;
    DLOG(INFO) << "sending to " << queue.front().endpoint
      << " datagram of size "
      << boost::asio::buffer_size(queue.front().bufs[1]);

    socket->async_send_to(
      queue.front().bufs,
      queue.front().endpoint,
      boost::bind(&SendQueue::sent, this, _1, _2,
                  EnabledShared::shared_from_this()));
    sending = true;
  }

  void sent(const boost::system::error_code& error,
            std::size_t size,
            boost::shared_ptr<SendQueue>) {
    if (error) {
      LOG(ERROR) << "sent failed with reason: " << error.message();
    }
    if (RaiseErrorOnDisconnect
        && (error == boost::system::errc::connection_refused)) {
      throw std::runtime_error("disconnected");
    }
    queue.front().callback(!error);
    queue.pop_front();
    sending = false;
    send();
  }

 public:
  explicit SendQueue(boost::shared_ptr<boost::asio::ip::udp::socket> socket)
    : socket(socket)
    , sending(false) {
  }

  void push(SendQueueItem item) {
    queue.push_back(item);
    send();
  }
};

class UdpClient : public Waypoint {
  boost::shared_ptr<Daemon> daemon;
  boost::shared_ptr<boost::asio::ip::udp::socket> socket;
  boost::shared_ptr<SendQueue<true> > queue;

  std::vector<Handler*> handlers;
  std::vector<char> recv_buffer_;
  bool connected;

  boost::asio::ip::udp::endpoint incoming_endpoint;
  boost::asio::ip::udp::endpoint server_endpoint;
  boost::posix_time::ptime last_received;

  virtual void time_expired() {
    queue->push(SendQueueItem(MsgType::Heartbeat, server_endpoint));
    DLOG(INFO) << "Client: Sending heartbeat to: " << server_endpoint;

    boost::posix_time::time_duration diff =
      boost::posix_time::second_clock::local_time() - last_received;
    if (diff.total_microseconds() > udp_disconnect_time_us) {
      throw std::runtime_error("disconnected");
    }
  }

  void start_receive() {
    socket->async_receive_from(boost::asio::buffer(recv_buffer_),
      incoming_endpoint,
      boost::bind(&UdpClient::handle_receive, this, _1, _2));
      DLOG(INFO) << "Client: waiting for message on "
        << socket->local_endpoint();
  }

  void handle_receive(const boost::system::error_code& error,
                      std::size_t bytes_transferred) {
    if (error) {
      LOG(ERROR) << "Client: Encountered a problem while receiving a message - "
        << error.message();
    } else {
      switch (recv_buffer_.front()) {
        case MsgType::Connect:
          handle_connected();
          break;
        case MsgType::Heartbeat:
          break;
        case MsgType::Message:
          handle_msg(bytes_transferred);
          break;
        case MsgType::Shutdown:
          handle_shutdown_received();
          break;
        default:
          LOG(ERROR) << "Client: Received an unexpected message type - "
            << recv_buffer_.at(0);
      }
      last_received = boost::posix_time::second_clock::local_time();
    }
    start_receive();
  }

  void handle_shutdown_received() {
    LOG(INFO) << "Client: Received shutdown from server";
    throw std::runtime_error("disconnected");
  }

  void connect_to_server() {
    queue->push(SendQueueItem(MsgType::Connect, server_endpoint));
    DLOG(INFO) << "Client: Sending connect to: " << server_endpoint;
  }

  void handle_msg(size_t bytes_transferred) {
    DLOG(INFO) << "Client: Received message from " << address()
      << " of size " << bytes_transferred-1;
    for (int i = 0; i < handlers.size(); ++i) {
      handlers[i]->received(
        &recv_buffer_.front() + 1, bytes_transferred - 1, this);
    }
  }

  void handle_connected() {
    VLOG(3) << "Client: connect received " << address();
    if (!connected) {
      create_timer(daemon, udp_tick_time_us,
                   boost::bind(&UdpClient::time_expired, this),
                   true);
      last_received = boost::posix_time::second_clock::local_time();
      connected = true;
    }
  }

 public:
  UdpClient(boost::shared_ptr<Daemon> daemon,
            boost::shared_ptr<boost::asio::ip::udp::socket> socket,
            boost::asio::ip::udp::endpoint endpoint,
            size_t max_packet_size)
            : daemon(daemon)
            , socket(socket)
            , queue(new SendQueue<true>(socket))
            , recv_buffer_(
                std::min(size_t(udp_max_datagram_size), max_packet_size + 1))
            , connected(false)
            , server_endpoint(endpoint) {
    VLOG(2) << "local: " << socket->local_endpoint()
      << " server: " << server_endpoint;
    connect_to_server();
    start_receive();
  }

  virtual ~UdpClient() {
    socket->cancel();
  }

  virtual void async_send(const char* buffer,
                          size_t size,
                          SentCallback callback) {
    queue->push(SendQueueItem(buffer, size, callback, server_endpoint));
  }

  virtual void register_receive_handler(Handler* handler) {
    handlers.push_back(handler);
  }

  virtual RemoteId id() const {
    return reinterpret_cast<RemoteId>(this);
  }

  virtual string address() const {
    return get_address(incoming_endpoint);
  }

  virtual bool guaranteed_comm() const {
    return false;
  }

  virtual size_t max_packet_size() const {
    return recv_buffer_.size() - 1;
  }
};

struct ClientEndpoint : Waypoint {
  const uint32_t                        buffer_size;
  boost::asio::ip::udp::endpoint        endpoint;
  boost::posix_time::ptime              last_received;
  std::vector<Waypoint::Handler*>       receive_handlers;
  SendQueue<false>*                     queue;

  ClientEndpoint(boost::shared_ptr<Daemon> daemon,
                 boost::asio::ip::udp::endpoint client_endpoint,
                 SendQueue<false>* queue,
                 size_t max_packet_size)
    : buffer_size(
        std::min(max_packet_size + 1, size_t(udp_max_datagram_size)))
    , endpoint(client_endpoint)
    , last_received(boost::posix_time::second_clock::local_time())
    , queue(queue) {
  }

  virtual void async_send(const char* buffer,
                          size_t size,
                          SentCallback callback) {
    queue->push(SendQueueItem(buffer, size, callback, endpoint));
    DLOG(INFO) << "Server: sending to: " << get_address(endpoint)
               << " buffer of size: " << size;
  }

  virtual void register_receive_handler(Handler* handler) {
    receive_handlers.push_back(handler);
  }

  virtual RemoteId id() const {
    return reinterpret_cast<RemoteId>(this);
  }

  virtual string address() const {
    return get_address(endpoint);
  }

  virtual bool guaranteed_comm() const {
    return false;
  }

  virtual size_t max_packet_size() const {
    return buffer_size - 1;
  }
};

template <bool UseMulticast>
class UdpServerCommunicatorImpl : public MultiWaypoint {
  typedef boost::unordered_map<string,
          boost::shared_ptr<ClientEndpoint> >                     Clients;
  typedef Clients::iterator                                       ClientIt;

  const uint32_t                                  buffer_size;
  boost::asio::ip::udp::endpoint                  local_endpoint;
  boost::asio::ip::udp::endpoint                  group_endpoint;
  boost::asio::ip::udp::endpoint                  incoming_client_endpoint;

  std::vector<Handler*>                           accept_handlers;
  std::vector<Waypoint::Handler*>                 receive_handlers;
  boost::shared_ptr<Daemon>                       daemon;

  boost::shared_ptr<boost::asio::ip::udp::socket> socket;
  Clients                                         clients;
  vector<char>                                    recv_buffer_;

  boost::shared_ptr<SendQueue<false> >            queue;


  virtual void time_expired() {
    for (ClientIt it = clients.begin(); it != clients.end(); ++it) {
      boost::posix_time::time_duration diff =
        boost::posix_time::second_clock::local_time()
        - it->second->last_received;
      if (diff.total_microseconds() > udp_disconnect_time_us)
        handle_disconnect(get_address(it->second->endpoint));
      else if (!UseMulticast) {
        queue->push(SendQueueItem(MsgType::Heartbeat, it->second->endpoint));
      }
    }
    if (UseMulticast) {
      queue->push(SendQueueItem(MsgType::Heartbeat, group_endpoint));
    }
  }

  void start_receive() {
    socket->async_receive_from(boost::asio::buffer(recv_buffer_),
      incoming_client_endpoint,
      boost::bind(&UdpServerCommunicatorImpl::handle_receive, this, _1, _2));
    DLOG(INFO) << "Server: waiting for message on "
      << socket->local_endpoint();
  }

  void handle_receive(const boost::system::error_code& error,
                      std::size_t bytes_transferred) {
    if (error) {
      LOG(ERROR) << "Server: Encountered a problem while receiving a message - "
        << error.message();
    } else {
      switch (recv_buffer_.front()) {
        case MsgType::Connect:
          handle_connect();
          break;
        case MsgType::Heartbeat:
          handle_received_heartbeat();
          break;
        case MsgType::Message:
          handle_msg(bytes_transferred);
          break;
        default:
          LOG(ERROR) << "Server: Received an unexpected message type - "
            << recv_buffer_.at(0);
      }
    }
    start_receive();
  }

  void handle_msg(std::size_t bytes_transferred) {
    ClientIt it = clients.find(get_address(incoming_client_endpoint));
    if (it == clients.end()) return;
    DLOG(INFO) << "Server: Received message from " <<
      get_address(incoming_client_endpoint)
      << "(" << it->second->id() << ")"
      << " of size " << bytes_transferred-1;
    for (int i = 0; i < it->second->receive_handlers.size(); ++i)
      it->second->receive_handlers[i]->received(
        &recv_buffer_.front() + 1, bytes_transferred - 1, it->second.get());
    it->second->last_received =
      boost::posix_time::second_clock::local_time();
  }

  void handle_received_heartbeat() {
    string addr = get_address(incoming_client_endpoint);
    ClientIt it = clients.find(addr);
    if (it != clients.end()) {
      it->second->last_received =
        boost::posix_time::second_clock::local_time();
      DLOG(INFO) << "Server: heartbeat received from " << addr;
    }
  }

  void handle_connect() {
    DLOG(INFO) << "Server: connect received";
    string addr = get_address(incoming_client_endpoint);
    ClientIt it = clients.find(addr);
    if (it == clients.end()) {
      it = clients.insert(
        std::make_pair(addr, boost::make_shared<ClientEndpoint>(
          daemon,
          incoming_client_endpoint,
          queue.get(),
          max_packet_size()))).first;
      LOG(INFO) << "Server: accepted client from address: " << addr
        << "(" << it->second->id() << ")";
      queue->push(SendQueueItem(MsgType::Connect, incoming_client_endpoint));
      for (int i = 0; i < accept_handlers.size(); ++i) {
        accept_handlers[i]->accepted(it->second);
      }
      for (int i = 0; i < receive_handlers.size(); ++i) {
        it->second->register_receive_handler(receive_handlers[i]);
      }
    } else {
      LOG(ERROR) << "Server: unexpected ack received from " << addr;
    }
  }

  void handle_disconnect(string addr) {
    LOG(INFO) << "[" << addr << "] client disconnected (" <<
      clients[addr]->id() << ")";
    for (int i = 0; i < accept_handlers.size(); ++i) {
      accept_handlers[i]->disconnected(clients[addr]->id());
    }
    clients.erase(addr);
  }

 public:
  UdpServerCommunicatorImpl(shared_ptr<Daemon> daemon,
                            boost::asio::ip::udp::endpoint local_endpoint,
                            boost::asio::ip::udp::endpoint group_endpoint,
                            size_t max_packet_size)
                    : buffer_size(
                        std::min(size_t(udp_max_datagram_size),
                                 max_packet_size + 1))
                    , local_endpoint(local_endpoint)
                    , group_endpoint(group_endpoint)
                    , daemon(daemon)
                    , socket(boost::make_shared<boost::asio::ip::udp::socket>(
                             boost::ref(get_io_service(daemon)),
                             local_endpoint))
                    , recv_buffer_(buffer_size)
                    , queue(new SendQueue<false>(socket)) {
    VLOG(2) << "local : " << local_endpoint
            << " group : " << group_endpoint;
    create_timer(daemon, udp_tick_time_us,
                 boost::bind(&UdpServerCommunicatorImpl::time_expired, this),
                 true);
    start_receive();
  }

  ~UdpServerCommunicatorImpl() {
    VLOG(3) << "Server destroyed " << get_address(local_endpoint);
  }

  virtual void register_peer_change_handler(Handler* handler) {
    accept_handlers.push_back(handler);
  }

  virtual void register_receive_handler(Waypoint::Handler* handler) {
    receive_handlers.push_back(handler);
    for (ClientIt it = clients.begin(); it != clients.end(); ++it) {
      it->second->register_receive_handler(handler);
    }
  }

  virtual void async_send(const char* buffer,
                          size_t size,
                          SentCallback callback) {
    if (clients.empty()) return;
    if (UseMulticast) {
      queue->push(
        SendQueueItem(buffer, size, callback, group_endpoint));
    } else {
      BroadcastCallback<SentCallback> broadcast_callback(callback);
      for (ClientIt it = clients.begin(); it != clients.end(); ++it) {
        queue->push(SendQueueItem(
          buffer, size, broadcast_callback, it->second->endpoint));
      }
    }
  }

  virtual RemoteId id() const {
    return reinterpret_cast<RemoteId>(this);
  }

  virtual string address() const {
    return get_address(local_endpoint);
  }

  virtual bool guaranteed_comm() const {
    return false;
  }

  virtual size_t max_packet_size() const {
    return buffer_size - 1;
  }
};

boost::asio::ip::udp::endpoint resolve(boost::shared_ptr<Daemon> daemon,
                                       std::string ip,
                                       std::string port) {
  using boost::asio::ip::udp;
  if (ip == "*") {
    udp::resolver::query query(udp::v4(), port);
    udp::resolver resolver(get_io_service(daemon));
    return *resolver.resolve(query);
  } else {
    udp::resolver::query query(ip, port);
    udp::resolver resolver(get_io_service(daemon));
    return *resolver.resolve(query);
  }
}

uint16_t to_port(std::string port) {
  return boost::lexical_cast<uint16_t>(port);
}

}  // namespace

boost::shared_ptr<Waypoint> configure_udp_client(
    boost::shared_ptr<Daemon> daemon,
    std::string ip,
    std::string port,
    std::string group_ip,
    std::string group_port,
    size_t max_buffer_size) {
  using boost::asio::ip::udp;

  boost::shared_ptr<Daemon> internal = create_communication_daemon();
  boost::shared_ptr<udp::socket> socket(
                      new udp::socket(get_io_service(internal)));
  socket->open(udp::v4());

  if (!group_ip.empty()) {
    boost::asio::ip::multicast::enable_loopback option(true);
    socket->set_option(udp::socket::reuse_address(true));
    socket->set_option(option);
    socket->bind(udp::endpoint(udp::v4(), to_port(group_port)));
    socket->set_option(boost::asio::ip::multicast::join_group(
          boost::asio::ip::address::from_string(group_ip)));
    LOG(INFO) << "Client: joined multicast group "
      << group_ip << ":" << group_port;
  }

  VLOG(1) << "configured udp client to server: "
    << resolve(internal, ip, port) << " and multicast group: "
    << group_ip << ":" << group_port;
  return configure_guaranteed_client(
    daemon, internal,
    boost::make_shared<UdpClient>(
      internal, socket, resolve(internal, ip, port), max_buffer_size));
}

boost::shared_ptr<MultiWaypoint> configure_udp_server(
    boost::shared_ptr<Daemon> communication_daemon,
    std::string ip,
    std::string port,
    std::string group_ip,
    std::string group_port,
    size_t max_buffer_size) {
  using boost::asio::ip::udp;

  boost::shared_ptr<Daemon> internal = create_communication_daemon();
  udp::endpoint local_endpoint = resolve(internal, ip, port);

  if (group_ip.size()) {
    udp::endpoint group_endpoint =
      resolve(internal, group_ip, group_port);
    VLOG(1) << "configured udp server on local endpoint: "
      << local_endpoint << " and multicast group: " << group_endpoint;
    return configure_guaranteed_server(
      communication_daemon, internal,
      boost::make_shared<UdpServerCommunicatorImpl<true> >(
        internal, local_endpoint, group_endpoint, max_buffer_size));
  }

  VLOG(1) << "configured udp server on local endpoint: " << local_endpoint;
  return configure_guaranteed_server(
      communication_daemon, internal,
    boost::make_shared<UdpServerCommunicatorImpl<false> >(
      internal, local_endpoint, udp::endpoint(), max_buffer_size));
}

}  // namespace internode
}  // namespace caffe

