#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/unordered_map.hpp>
#include <string>
#include <utility>
#include <vector>
#include "caffe/internode/communication.hpp"
#include "caffe/internode/configuration.hpp"

namespace caffe {
namespace internode {

class Daemon {
 public:
    boost::asio::io_service io_service;
};

void run(shared_ptr<Daemon> daemon) {
  daemon->io_service.run();
}

void run_one(shared_ptr<Daemon> daemon) {
  daemon->io_service.run_one();
}

void poll_one(shared_ptr<Daemon> daemon) {
  daemon->io_service.poll_one();
}

void poll(shared_ptr<Daemon> daemon) {
  daemon->io_service.poll();
}

namespace {

string get_address(boost::asio::ip::tcp::socket* socket) {
  try {
    std::string ip = socket->remote_endpoint().address().to_string();
    unsigned short port = socket->remote_endpoint().port();

    return "tcp://" + ip + ":" + boost::lexical_cast<string>(port);
  } catch (...) {
    return "tcp://...";
  }
}

void null_disconnect_handler(string addr) {
  throw std::runtime_error("[" + addr + "] client disconnected");
}

class SingleClient : public Waypoint {
  boost::shared_ptr<boost::asio::ip::tcp::socket> socket;
  typedef boost::function<void(string)> DisconnectHandler;
  DisconnectHandler disconnect_handler;

  std::vector<Handler*> handlers;
  size_t size_buffer;
  std::vector<char> buffer;
  string address;
  boost::recursive_mutex send_mtx;

  void handle_size(const boost::system::error_code& ec, size_t size) {
    // size_buffer = ntohl(size_buffer);

    if (ec) {
      LOG(ERROR) << "[" << address << "] "
                 << "received error on receiving size: " << ec.message()
                 << " " << size << ", client is closed";
      disconnect_handler(address);
      return;
    }
    if (size != sizeof(size_buffer)) {
      LOG(ERROR) << "[" << address << "] "
                 << "received error on receiving size: " << size;
      return;
    }
    CHECK(size_buffer < 500 * 1024 * 1024)
      << "[" << address << "] size buffer is too big: " << size_buffer;
    if (buffer.size() < size_buffer) {
      buffer.resize(size_buffer);
    }
    using boost::asio::async_read;
    using boost::asio::transfer_exactly;
    using boost::bind;
    async_read(
      *socket,
      boost::asio::buffer(&buffer.front(), size_buffer),
      transfer_exactly(size_buffer),
      bind(&SingleClient::handle_msg, this, _1, _2));
  }

  void handle_msg(const boost::system::error_code& ec, size_t size) {
    if (ec) {
      LOG(ERROR) << "[" << address << "] "
                 << "received error on receiving size: " << ec.message()
                 << " " << size << ", client is closed";
      disconnect_handler(address);
      return;
    }
    if (size != size_buffer) {
      LOG(ERROR) << "[" << address << "] "
                 << "received error on receiving size: " << size;
      return;
    }
    for (int i = 0; i < handlers.size(); ++i) {
      handlers[i]->received(&buffer.front(), size_buffer, get_id());
    }
    async_receive();
  }

  void async_receive() {
    boost::asio::async_read(
      *socket,
      boost::asio::buffer(&size_buffer, sizeof(size_buffer)),
      boost::asio::transfer_exactly(sizeof(size_buffer)),
      boost::bind(&SingleClient::handle_size, this, _1, _2));
  }

 public:
  SingleClient(boost::shared_ptr<boost::asio::ip::tcp::socket> socket,
               DisconnectHandler disconnect_handler)
      : socket(socket)
      , disconnect_handler(disconnect_handler)
      , address(get_address(socket.get())) {
    async_receive();
  }

  ~SingleClient() {
    LOG(INFO) << "client " << address << " destroyed";
  }

  virtual void send(const char* buffer, size_t size) {
    boost::recursive_mutex::scoped_lock lock(send_mtx);
    using boost::asio::write;
    size_t sent_size = size;
    boost::system::error_code ec1, ec2;
    DLOG(INFO) << "sending to: " << address << " buffer of size: " << size;
    write(*socket, boost::asio::buffer(&sent_size, sizeof(sent_size)), ec1);
    write(*socket, boost::asio::buffer(buffer, size), ec2);
    if (ec1 || ec2) {
      LOG(ERROR) << "sending failed with error: " << ec1.message() << " "
        << ec2.message() << " " << socket->is_open();
      return;
    }
    DLOG(INFO) << "sent to: " << address << " buffer of size: " << size;
  }

  virtual void register_receive_handler(Handler* handler) {
    handlers.push_back(handler);
  }

  virtual void shutdown() {
    boost::system::error_code ec;
    socket->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    if (ec) {
      LOG(ERROR) << "shutdown failed with error: " << ec.message();
    }
  }

  virtual void close() {
    if (socket->is_open()) {
      socket->cancel();
      socket->close();
    }
  }

  virtual RemoteId get_id() {
    return reinterpret_cast<RemoteId>(this);
  }
};

class ServerCommunicatorImpl : public MultiWaypoint {
  typedef boost::shared_ptr<boost::asio::ip::tcp::socket> SharedTcpSocket;
  typedef boost::shared_ptr<SingleClient> Client;
  typedef boost::unordered_map<string, Client> Clients;
  typedef Clients::iterator ClientIt;

  boost::shared_ptr<Daemon>       daemon;
  boost::asio::ip::tcp::endpoint  endpoint;
  boost::asio::ip::tcp::acceptor  acceptor;
  SharedTcpSocket                 new_client_socket;
  Clients                         clients;

  std::vector<Handler*>           accept_handlers;
  std::vector<Waypoint::Handler*> receive_handlers;
  boost::recursive_mutex                    mtx;

  void start_accept() {
    boost::recursive_mutex::scoped_lock lock(mtx);
    acceptor.async_accept(
      *new_client_socket,
      boost::bind(&ServerCommunicatorImpl::handle_accept, this, _1));
  }

  void handle_accept(const boost::system::error_code& error) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    string address = get_address(new_client_socket.get());
    LOG(INFO) << "accepted client from address: " << address;
    clients[address].reset(new SingleClient(
      new_client_socket,
      bind(&ServerCommunicatorImpl::handle_disconnect, this, _1)));
    for (int i = 0; i < receive_handlers.size(); ++i) {
      clients[address]->register_receive_handler(receive_handlers[i]);
    }
    new_client_socket =
      boost::make_shared<boost::asio::ip::tcp::socket>(
        boost::ref(daemon->io_service));

    for (int i = 0; i < accept_handlers.size(); ++i) {
      accept_handlers[i]->accepted(clients[address]->get_id());
    }

    start_accept();
  }

  void handle_disconnect(string addr) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    LOG(INFO) << "[" << addr << "] client disconnected (" <<
      clients[addr]->get_id() << ")";
    for (int i = 0; i < accept_handlers.size(); ++i) {
      accept_handlers[i]->disconnected(clients[addr]->get_id());
    }

    clients.erase(addr);
  }

 public:
  ServerCommunicatorImpl(
          boost::shared_ptr<Daemon> daemon,
          std::string port)
      : daemon(daemon)
      , endpoint(boost::asio::ip::tcp::v4(),
                 boost::lexical_cast<uint16_t>(port))
      , acceptor(daemon->io_service, endpoint)
      , new_client_socket(
          boost::make_shared<boost::asio::ip::tcp::socket>(
            boost::ref(daemon->io_service))) {
    start_accept();
  }

  virtual void register_peer_change_handler(Handler* handler) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    accept_handlers.push_back(handler);
  }

  virtual void send(const char* buffer, size_t size) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    for (ClientIt it = clients.begin(); it != clients.end(); ++it) {
      it->second->send(buffer, size);
    }
  }
  virtual void send_to(const char* buffer, size_t size, RemoteId id) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    for (ClientIt it = clients.begin(); it != clients.end(); ++it) {
      if (it->second->get_id() == id) {
        it->second->send(buffer, size);
      }
    }
  }
  virtual void register_receive_handler(Waypoint::Handler* handler) {
    boost::recursive_mutex::scoped_lock lock(mtx);
    for (ClientIt it = clients.begin(); it != clients.end(); ++it) {
      it->second->register_receive_handler(handler);
    }
    receive_handlers.push_back(handler);
  }
  virtual void close() {
    boost::recursive_mutex::scoped_lock lock(mtx);
    for (ClientIt it = clients.begin(); it != clients.end(); ++it) {
      it->second->close();
    }
    acceptor.close();
  }
  virtual void shutdown() {
    boost::recursive_mutex::scoped_lock lock(mtx);
    for (ClientIt it = clients.begin(); it != clients.end(); ++it) {
      it->second->shutdown();
    }
  }
};

std::pair<string, string> extract(string address) {
  string tcp_protcol = "tcp://";
  size_t protocol_pos = address.find_first_of(tcp_protcol);
  size_t port_pos = address.find_first_of(":");
  port_pos = address.find_first_of(":", port_pos + 1);
  if ((protocol_pos != 0) || (port_pos == std::string::npos)) {
    LOG(ERROR) << "unrecognized address: " << address
      << ", expected format is: `tcp://*:80`";
    throw std::runtime_error("unsupported protocol");
  }

  string ip = address.substr(tcp_protcol.size(), port_pos - tcp_protcol.size());
  string port = address.substr(port_pos + 1);
  return std::make_pair(ip, port);
}

}  // namespace

boost::shared_ptr<Daemon> create_communication_daemon() {
  return boost::make_shared<Daemon>();
}

boost::shared_ptr<Waypoint> configure_client(
    boost::shared_ptr<Daemon> daemon,
    std::string address) {
  string ip = extract(address).first;
  string port = extract(address).second;

  boost::asio::ip::tcp::resolver::query query(ip, port);
  boost::asio::ip::tcp::resolver resolver(daemon->io_service);
  boost::asio::ip::tcp::resolver::iterator endpoint_it(resolver.resolve(query));
  boost::shared_ptr<boost::asio::ip::tcp::socket> socket
    (new boost::asio::ip::tcp::socket(daemon->io_service));
  try {
    boost::asio::connect(*socket, endpoint_it);
  } catch (std::runtime_error& error) {
    LOG(INFO) << "connect failed: " << error.what();
  } catch (...) {
    LOG(INFO) << "connect failed: ...";
  }

  return boost::make_shared<SingleClient>(socket, null_disconnect_handler);
}

boost::shared_ptr<MultiWaypoint> configure_server(
    boost::shared_ptr<Daemon> communication_daemon,
    string address) {
  string port = extract(address).second;
  return boost::make_shared<ServerCommunicatorImpl>(
      communication_daemon, port);
}

bool is_remote_address(std::string str) {
  static string tcp_prefix = "tcp://";
  return ((str.size() > tcp_prefix.size())
      && (str.substr(0, tcp_prefix.size()) == tcp_prefix));
}

}  // namespace internode
}  // namespace caffe

