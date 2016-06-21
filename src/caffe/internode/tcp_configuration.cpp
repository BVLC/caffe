#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/unordered_map.hpp>
#include <algorithm>
#include <deque>
#include <string>
#include <utility>
#include <vector>
#include "caffe/internode/broadcast_callback.hpp"
#include "caffe/internode/communication.hpp"
#include "caffe/internode/configuration.hpp"

namespace caffe {
namespace internode {

extern boost::asio::io_service& get_io_service(boost::shared_ptr<Daemon>);

namespace {

typedef uint64_t MsgSize;

string get_address(const boost::asio::ip::tcp::endpoint&  endpoint) {
  try {
    std::string ip = endpoint.address().to_string();
    unsigned short port = endpoint.port();

    return "tcp://" + ip + ":" + boost::lexical_cast<string>(port);
  } catch (...) {
    return "tcp://...";
  }
}

string get_address(const boost::asio::ip::tcp::socket& socket) {
  return get_address(socket.remote_endpoint());
}

void null_disconnect_handler(string addr) {
  throw std::runtime_error("[" + addr + "] client disconnected");
}

struct SendQueueItem {
  boost::shared_ptr<uint64_t> size;
  Waypoint::SentCallback callback;
  boost::array<boost::asio::const_buffer, 2> bufs;

  SendQueueItem(const char* buffer,
                uint64_t size,
                Waypoint::SentCallback callback)
    : size(new uint64_t(size))
    , callback(callback) {
    bufs[0] = boost::asio::buffer(this->size.get(), sizeof(uint64_t));
    bufs[1] = boost::asio::buffer(buffer, size);
  }
};

class SendQueue : public boost::enable_shared_from_this<SendQueue> {
  std::deque<SendQueueItem> queue;
  boost::shared_ptr<boost::asio::ip::tcp::socket> socket;
  bool sending;
  boost::mutex mtx;

  void send() {
    if (sending) return;
    if (queue.empty()) return;
    DLOG(INFO) << "sending tcp packet of size "
      << boost::asio::buffer_size(queue.front().bufs[1]);

    sending = true;
    boost::asio::async_write(
      *socket,
      queue.front().bufs,
      boost::bind(&SendQueue::sent, this, _1, _2, shared_from_this()));
  }

  void sent(const boost::system::error_code& error,
            std::size_t size,
            boost::shared_ptr<SendQueue>) {
    Waypoint::SentCallback callback;
    {
      boost::mutex::scoped_lock lock(mtx);
      if (error) {
        LOG(ERROR) << "sent failed with reason: " << error.message();
      }
      CHECK(!queue.empty());
      callback = queue.front().callback;
      queue.pop_front();
    }
    callback(!error);
    {
      boost::mutex::scoped_lock lock(mtx);
      sending = false;
      send();
    }
  }

 public:
  explicit SendQueue(boost::shared_ptr<boost::asio::ip::tcp::socket> socket)
    : socket(socket)
    , sending(false) {
  }

  void push(SendQueueItem item) {
    boost::mutex::scoped_lock lock(mtx);
    queue.push_back(item);
    send();
  }
};

class SingleClient : public boost::enable_shared_from_this<SingleClient>
                   , public Waypoint {
  const MsgSize buffer_size;
  boost::shared_ptr<boost::asio::ip::tcp::socket> socket;
  typedef boost::function<void(string)> DisconnectHandler;
  DisconnectHandler disconnect_handler;

  std::vector<Handler*> handlers;
  MsgSize size_buffer;
  std::vector<char> buffer;
  string address_;
  boost::recursive_mutex send_mtx;
  boost::shared_ptr<SendQueue> queue;

  void handle_size(const boost::system::error_code& ec,
                   size_t size,
                   boost::shared_ptr<SingleClient> shared_this) {
    // size_buffer = ntohl(size_buffer);
    if (ec) {
      LOG(ERROR) << "[" << address() << "] "
                 << "received error on receiving size: " << ec.message()
                 << " " << size << ", client is closed";
      disconnect_handler(address());
      return;
    }
    if (size != sizeof(size_buffer)) {
      LOG(ERROR) << "[" << address() << "] "
                 << "received error on receiving size: " << size;
      return;
    }
    DLOG(INFO) << "receiving msg of size: " << size_buffer;
    CHECK(size_buffer < 500 * 1024 * 1024)
      << "[" << address() << "] size buffer is too big: " << size_buffer;
    if (buffer.size() < size_buffer) {
      buffer.resize(size_buffer);
    }
    DLOG(INFO) << "expecting buffer of size: " << size_buffer;
    using boost::asio::async_read;
    using boost::asio::transfer_exactly;
    using boost::bind;
    async_read(
      *socket,
      boost::asio::buffer(&buffer.front(), size_buffer),
      transfer_exactly(size_buffer),
      bind(&SingleClient::handle_msg, this, _1, _2, shared_from_this()));
  }

  void handle_msg(const boost::system::error_code& ec,
                  size_t size,
                  boost::shared_ptr<SingleClient> shared_this) {
    if (ec) {
      LOG(ERROR) << "[" << address() << "] "
                 << "received error on receiving size: " << ec.message()
                 << " " << size << ", client is closed";
      disconnect_handler(address());
      return;
    }
    if (size != size_buffer) {
      LOG(ERROR) << "[" << address() << "] "
                 << "received error on receiving size: " << size;
      return;
    }
    for (int i = 0; i < handlers.size(); ++i) {
      handlers[i]->received(&buffer.front(), size_buffer, this);
    }
    async_receive();
  }

  void async_receive() {
    boost::asio::async_read(
      *socket,
      boost::asio::buffer(&size_buffer, sizeof(size_buffer)),
      boost::asio::transfer_exactly(sizeof(size_buffer)),
      boost::bind(
        &SingleClient::handle_size, this, _1, _2, shared_from_this()));
  }

 public:
  SingleClient(boost::shared_ptr<boost::asio::ip::tcp::socket> socket,
               DisconnectHandler disconnect_handler,
               uint32_t max_packet_size)
      : buffer_size(
          std::min(max_packet_size + sizeof(MsgSize), 1024 * 1024 * 1024lu))
      , socket(socket)
      , disconnect_handler(disconnect_handler)
      , address_(get_address(*socket))
      , queue(new SendQueue(socket)) {
  }

  virtual ~SingleClient() {
    socket->cancel();
    LOG(INFO) << "client " << address() << " destroyed";
  }

  void start() {
    async_receive();
  }

  virtual void async_send(const char* buffer,
                          size_t size,
                          SentCallback callback) {
    DLOG(INFO) << "sending to: " << address() << " buffer of size: " << size;
    boost::recursive_mutex::scoped_lock lock(send_mtx);
    queue->push(SendQueueItem(buffer, size, callback));
  }

  virtual void register_receive_handler(Handler* handler) {
    handlers.push_back(handler);
  }

  virtual RemoteId id() const {
    return reinterpret_cast<RemoteId>(this);
  }

  virtual string address() const {
    return address_;
  }

  virtual bool guaranteed_comm() const {
    return true;
  }

  virtual size_t max_packet_size() const {
    return buffer_size - sizeof(MsgSize);
  }
};

class ServerCommunicatorImpl : public MultiWaypoint {
  typedef boost::shared_ptr<boost::asio::ip::tcp::socket> SharedTcpSocket;
  typedef boost::shared_ptr<SingleClient> Client;
  typedef boost::unordered_map<string, Client> Clients;
  typedef Clients::iterator ClientIt;

  const MsgSize                   buffer_size;
  boost::shared_ptr<Daemon>       daemon;
  boost::asio::ip::tcp::endpoint  endpoint;
  boost::asio::ip::tcp::acceptor  acceptor;
  SharedTcpSocket                 new_client_socket;
  Clients                         clients;

  std::vector<Handler*>           accept_handlers;
  std::vector<Waypoint::Handler*> receive_handlers;
  boost::recursive_mutex          mtx;

  void start_accept() {
    boost::recursive_mutex::scoped_lock lock(mtx);
    acceptor.async_accept(
      *new_client_socket,
      boost::bind(&ServerCommunicatorImpl::handle_accept, this, _1));
  }

  void handle_accept(const boost::system::error_code& error) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    string address = get_address(*new_client_socket);
    LOG(INFO) << "accepted client from address: " << address;
    clients[address].reset(new SingleClient(
      new_client_socket,
      bind(&ServerCommunicatorImpl::handle_disconnect, this, _1),
      max_packet_size()));
    clients[address]->start();
    for (int i = 0; i < receive_handlers.size(); ++i) {
      clients[address]->register_receive_handler(receive_handlers[i]);
    }
    new_client_socket =
      boost::make_shared<boost::asio::ip::tcp::socket>(
        boost::ref(get_io_service(daemon)));

    for (int i = 0; i < accept_handlers.size(); ++i) {
      accept_handlers[i]->accepted(clients[address]);
    }

    start_accept();
  }

  void handle_disconnect(string addr) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    LOG(INFO) << "[" << addr << "] client disconnected (" <<
      clients[addr]->id() << ")";
    for (int i = 0; i < accept_handlers.size(); ++i) {
      accept_handlers[i]->disconnected(clients[addr]->id());
    }

    clients.erase(addr);
  }

 public:
  ServerCommunicatorImpl(
          boost::shared_ptr<Daemon> daemon,
          std::string port,
          uint32_t max_packet_size)
      : buffer_size(
          std::min(max_packet_size + sizeof(MsgSize), 1024 * 1024 * 1024lu))
      , daemon(daemon)
      , endpoint(boost::asio::ip::tcp::v4(),
                 boost::lexical_cast<uint16_t>(port))
      , acceptor(get_io_service(daemon), endpoint)
      , new_client_socket(
          boost::make_shared<boost::asio::ip::tcp::socket>(
            boost::ref(get_io_service(daemon)))) {
    start_accept();
  }

  virtual void register_peer_change_handler(Handler* handler) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    accept_handlers.push_back(handler);
  }

  virtual void async_send(const char* buffer,
                          size_t size,
                          SentCallback callback) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    if (clients.empty()) return;
    BroadcastCallback<SentCallback> broadcast_callback(callback);
    for (ClientIt it = clients.begin(); it != clients.end(); ++it) {
      it->second->async_send(buffer, size, broadcast_callback);
    }
  }
  virtual void register_receive_handler(Waypoint::Handler* handler) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    for (ClientIt it = clients.begin(); it != clients.end(); ++it) {
      it->second->register_receive_handler(handler);
    }
    receive_handlers.push_back(handler);
  }

  virtual RemoteId id() const {
    return reinterpret_cast<RemoteId>(this);
  }

  virtual string address() const {
    return get_address(endpoint);
  }

  virtual bool guaranteed_comm() const {
    return true;
  }

  virtual size_t max_packet_size() const {
    return buffer_size - sizeof(MsgSize);
  }
};

}  // namespace

boost::shared_ptr<Waypoint> configure_tcp_client(
    boost::shared_ptr<Daemon> daemon,
    std::string ip,
    std::string port,
    size_t max_buffer_size) {

  boost::asio::ip::tcp::resolver::query query(ip, port);
  boost::asio::ip::tcp::resolver resolver(get_io_service(daemon));
  boost::asio::ip::tcp::resolver::iterator endpoint_it(resolver.resolve(query));
  boost::shared_ptr<boost::asio::ip::tcp::socket> socket
    (new boost::asio::ip::tcp::socket(get_io_service(daemon)));
  try {
    boost::asio::connect(*socket, endpoint_it);
  } catch (std::runtime_error& error) {
    LOG(INFO) << "connect failed: " << error.what();
  } catch (...) {
    LOG(INFO) << "connect failed: ...";
  }

  boost::shared_ptr<SingleClient> ret(
    new SingleClient(socket, null_disconnect_handler, max_buffer_size));
  ret->start();
  return ret;
}

boost::shared_ptr<MultiWaypoint> configure_tcp_server(
    boost::shared_ptr<Daemon> communication_daemon,
    std::string port,
    size_t max_buffer_size) {
  return boost::make_shared<ServerCommunicatorImpl>(
    communication_daemon, port, max_buffer_size);
}

}  // namespace internode
}  // namespace caffe

