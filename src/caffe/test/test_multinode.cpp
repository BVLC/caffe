#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/make_shared.hpp>
#include <gtest/gtest.h>
#include <string>
#include "caffe/internode/configuration.hpp"

namespace caffe {
namespace internode {

class ConnectionTest : public ::testing::Test
                     , public MultiWaypoint::Handler
                     , public Waypoint::Handler {
 public:
  boost::shared_ptr<Daemon> daemon;
  boost::shared_ptr<Waypoint> accepted_client;
  bool test_finished;
  string short_msg;
  string long_msg;
  string expected_msg;

  ConnectionTest()
    : daemon(create_communication_daemon())
    , test_finished(false)
    , short_msg("This message is sent from server to client during the test")
    , long_msg(65505, 'A') {
  }

  void sent(bool result) {
    ASSERT_TRUE(result);
  }

  void timer_expired() {
    throw std::runtime_error("timer expired");
  }

  void received(char* msg, size_t size, Waypoint*) {
    ASSERT_EQ(std::string(msg, size - 1), expected_msg);
    test_finished = true;
  }

  void accepted(shared_ptr<Waypoint> waypoint) {
    accepted_client = waypoint;
  }
  void disconnected(RemoteId) {
  }

  void check_connection(string address, string msg) {
    expected_msg = msg;

    boost::shared_ptr<MultiWaypoint> server =
      configure_server(daemon, address, UINT_MAX);
    server->register_peer_change_handler(this);

    boost::shared_ptr<Waypoint> client =
      configure_client(daemon, address, UINT_MAX);
    client->register_receive_handler(this);

    create_timer(
      daemon, 5e+6, boost::bind(&ConnectionTest::timer_expired, this), false);

    while (!accepted_client) {
      poll_one(daemon);
    }

    accepted_client->async_send(msg.c_str(), msg.size() + 1,
      boost::bind(&ConnectionTest::sent, this, _1));

    while (!test_finished) {
      poll_one(daemon);
    }
  }

  void check_multicast(string address, string msg) {
    expected_msg = msg;

    boost::shared_ptr<MultiWaypoint> server =
      configure_server(daemon, address, UINT_MAX);
    server->register_peer_change_handler(this);

    boost::shared_ptr<Waypoint> client =
      configure_client(daemon, address, UINT_MAX);
    client->register_receive_handler(this);

    create_timer(
      daemon, 5e+6, boost::bind(&ConnectionTest::timer_expired, this), false);

    while (!accepted_client) {
      poll_one(daemon);
    }

      server->async_send(msg.c_str(), msg.size() + 1,
      boost::bind(&ConnectionTest::sent, this, _1));

    while (!test_finished) {
      poll_one(daemon);
    }
  }
};  // class ConnectionTest

TEST_F(ConnectionTest, DISABLED_UdpConnect) {
  EXPECT_NO_FATAL_FAILURE(check_connection("udp://127.0.0.1:6969", short_msg));
}

TEST_F(ConnectionTest, DISABLED_UdpLargeString) {
  EXPECT_NO_FATAL_FAILURE(check_connection("udp://127.0.0.1:6969", long_msg));
}

TEST_F(ConnectionTest, DISABLED_TcpConnect) {
  EXPECT_NO_FATAL_FAILURE(check_connection("tcp://127.0.0.1:6969", short_msg));
}

TEST_F(ConnectionTest, DISABLED_UdpMulticast) {
  EXPECT_NO_FATAL_FAILURE(check_multicast("udp://127.0.0.1:6969;224.0.0.0:6970",
                                              short_msg));
}

}  // namespace internode
}  // namespace caffe
