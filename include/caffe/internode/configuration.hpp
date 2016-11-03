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

}  // namespace internode
}  // namespace caffe

#endif  // CAFFE_INTERNODE_CONFIGURATION_H_

