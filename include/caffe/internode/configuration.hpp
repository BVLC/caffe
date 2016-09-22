#ifndef CAFFE_INTERNODE_CONFIGURATION_H_
#define CAFFE_INTERNODE_CONFIGURATION_H_

#include <string>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>

#include "communication.hpp"

namespace caffe {
namespace internode {

typedef boost::function<void()> TimerCallback;

class Daemon {
  typedef boost::asio::deadline_timer Timer;
  typedef boost::posix_time::microseconds Microseconds;
  boost::asio::io_service io_service;

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
  void create_timer(uint64_t duration, TimerCallback callback, bool repeat) {
    shared_ptr<Timer> timer(new Timer(io_service, Microseconds(duration)));
    timer->async_wait(
      boost::bind(&Daemon::expired, this, _1, callback, timer, repeat,
        Microseconds(duration)));
  }

  boost::asio::io_service& get_io_service() {
    return io_service;
  }

  void run_one() {
    io_service.run_one();
  }

  void poll_one() {
    io_service.poll_one();
  }
};

inline void
run_one(shared_ptr<Daemon> daemon) {
  daemon->run_one();
}

inline void
poll_one(shared_ptr<Daemon> daemon) {
  daemon->poll_one();
}

inline boost::asio::io_service&
get_io_service(boost::shared_ptr<Daemon> daemon) {
  return daemon->get_io_service();
}

inline void
create_timer(boost::shared_ptr<Daemon> daemon,
                  uint64_t duration_microseconds,
                  TimerCallback callback,
                  bool repeat) {
  daemon->create_timer(duration_microseconds, callback, repeat);
}

inline boost::shared_ptr<Daemon>
create_communication_daemon() {
  return boost::make_shared<Daemon>();
}

boost::shared_ptr<Waypoint> configure_client(
    boost::shared_ptr<Daemon> communication_daemon,
    std::string address,
    size_t max_buffer_size);
boost::shared_ptr<MultiWaypoint> configure_server(
    boost::shared_ptr<Daemon> communication_daemon,
    std::string address,
    size_t max_buffer_size);

bool is_remote_address(std::string str);

}  // namespace internode
}  // namespace caffe

#endif  // CAFFE_INTERNODE_CONFIGURATION_H_

