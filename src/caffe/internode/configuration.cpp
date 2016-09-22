/* */

#include <string>
#include <vector>
#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>

#include "caffe/internode/communication.hpp"
#include "caffe/internode/configuration.hpp"
#include "caffe/internode/mpi_configuration.hpp"

namespace caffe {
namespace internode {

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

