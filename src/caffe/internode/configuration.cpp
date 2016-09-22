#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>
#include <string>
#include <vector>
#include "caffe/internode/communication.hpp"
#include "caffe/internode/configuration.hpp"
#include "caffe/internode/mpi_configuration.hpp"

namespace caffe {
namespace internode {

class Daemon {
  typedef boost::asio::deadline_timer Timer;
  typedef boost::posix_time::microseconds Microseconds;

  void expired(const boost::system::error_code& error,
               TimerCallback callback,
               shared_ptr<Timer> timer,
               bool repeat,
               Microseconds duration) {
    if (!error) callback();
    if (repeat) {
      timer->expires_at(timer->expires_at() + duration);
      timer->async_wait(
        boost::bind(&Daemon::expired, this, _1, callback, timer, repeat,
          duration));
    }
  }

 public:
  boost::asio::io_service io_service;

  void create_timer(uint64_t duration, TimerCallback callback, bool repeat) {
    shared_ptr<Timer> timer(new Timer(io_service, Microseconds(duration)));
    timer->async_wait(
      boost::bind(&Daemon::expired, this, _1, callback, timer, repeat,
        Microseconds(duration)));
  }
};

void run_one(shared_ptr<Daemon> daemon) {
  daemon->io_service.run_one();
}

void poll_one(shared_ptr<Daemon> daemon) {
  daemon->io_service.poll_one();
}

boost::asio::io_service& get_io_service(boost::shared_ptr<Daemon> daemon) {
  return daemon->io_service;
}

void create_timer(boost::shared_ptr<Daemon> daemon,
                  uint64_t duration_microseconds,
                  TimerCallback callback,
                  bool repeat) {
  daemon->create_timer(duration_microseconds, callback, repeat);
}

boost::shared_ptr<Daemon> create_communication_daemon() {
  return boost::make_shared<Daemon>();
}

struct Protocol {
  enum Type {
    NONE, UDP, TCP, MPI
  };
};

struct AddressInfo {
  Protocol::Type protocol;
  string ip;
  string port;
  string group_ip;
  string group_port;

  explicit AddressInfo(Protocol::Type protocol,
                       string ip = "",
                       string port = "",
                       string group_ip = "",
                       string group_port = "")
          : protocol(protocol)
          , ip(ip)
          , port(port)
          , group_ip(group_ip)
          , group_port(group_port) {
  }
};

AddressInfo extract(string address) {
  static const string separator = "://";
  static const string tcp_prefix = "tcp" + separator;
  static const string udp_prefix = "udp" + separator;
  static const string mpi_prefix = "mpi" + separator;
  size_t tcp_protocol_pos = address.find(tcp_prefix);
  size_t udp_protocol_pos = address.find(udp_prefix);
  size_t mpi_protocol_pos = address.find(mpi_prefix);
  if ((tcp_protocol_pos != 0)
      && (udp_protocol_pos != 0)
      && (mpi_protocol_pos != 0)) {
    return AddressInfo(Protocol::NONE);
  }

  size_t port_pos = address.find_first_of(":");
         port_pos = address.find_first_of(":", port_pos + 1);
  size_t sep_pos = address.find("://");
  size_t ip_pos = sep_pos + separator.size();
  size_t group_ip_pos = address.find(";");
  size_t group_port_pos = address.find_last_of(":");

  Protocol::Type protocol = Protocol::TCP;
  if (udp_protocol_pos == 0) protocol = Protocol::UDP;
  if (mpi_protocol_pos == 0) {
    return AddressInfo(Protocol::MPI, address.substr(ip_pos));
  }
if(group_ip_pos == std::string::npos) {
    string ip = address.substr(ip_pos, port_pos - ip_pos);
    string port = address.substr(port_pos + 1);
    return AddressInfo(protocol, ip, port);
  } else {
    string ip = address.substr(ip_pos, port_pos - ip_pos);
    string port = address.substr(port_pos + 1, group_ip_pos - 1 - port_pos);
    string group_ip = address.substr(
      group_ip_pos + 1, group_port_pos - group_ip_pos - 1);
    string group_port = address.substr(group_port_pos + 1);
    return AddressInfo(protocol, ip, port, group_ip, group_port);
  }
}

boost::shared_ptr<MultiWaypoint> configure_server(
    boost::shared_ptr<Daemon> communication_daemon,
    string address,
    size_t max_buffer_size) {

  AddressInfo info = extract(address);
  switch (info.protocol) {
    case Protocol::MPI:
      return configure_mpi_server(
        communication_daemon, info.ip, max_buffer_size);
    default:
      LOG(ERROR) << "unrecognized address: " << address
        << ", expected format is: `tcp://*:80` or `udp://*:777` "
        << "or `mpi://server_name`";
      throw std::runtime_error("invalid address");
  }
}

boost::shared_ptr<Waypoint> configure_client(
    boost::shared_ptr<Daemon> communication_daemon,
    string address,
    size_t max_buffer_size) {

  AddressInfo info = extract(address);
  switch (info.protocol) {
    case Protocol::MPI:
      return configure_mpi_client(
              communication_daemon, info.ip, max_buffer_size);
    default:
      LOG(ERROR) << "unrecognized address: " << address
        << ", expected format is: `tcp://*:80` or `udp://*:777` "
        << "or `mpi://server_name`";
      throw std::runtime_error("invalid address");
  }
}

bool is_remote_address(std::string str) {
  return extract(str).protocol != Protocol::NONE;
}

}  // namespace internode
}  // namespace caffe

