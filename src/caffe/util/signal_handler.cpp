#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <glog/logging.h>

#include "caffe/util/signal_handler.h"

#ifdef SIGHUP
#define SUPPORTED_SIGNALS SIGINT, SIGHUP
#else
#define SUPPORTED_SIGNALS SIGINT
#endif

namespace {
  static boost::asio::io_service io;
  static boost::asio::signal_set signals(io, SUPPORTED_SIGNALS);
  static bool got_signal = false;
  static int received_signal;
  static bool already_hooked_up = false;

  void handle_signal(const boost::system::error_code& error, int signal) {
    if (!error) {
      got_signal = true;
      received_signal = signal;
      signals.async_wait(handle_signal);
    } else if (error != boost::asio::error::operation_aborted) {
        LOG(FATAL) << "Signal handling error: " << error;
    }
  }

  void HookupHandler() {
    if (already_hooked_up) {
      LOG(FATAL) << "Tried to hookup signal handlers more than once.";
    }
    already_hooked_up = true;
    signals.async_wait(handle_signal);
  }

  // Set the signal handlers to the default.
  void UnhookHandler() {
    if (already_hooked_up) {
      signals.cancel();
      already_hooked_up = false;
    }
  }

  // Return true iff a signal was received.
  bool GotSignal() {
    io.poll();
    bool result = got_signal;
    got_signal = false;
    return result;
  }

}  // namespace

namespace caffe {

SignalHandler::SignalHandler(SolverAction::Enum SIGINT_action,
                             SolverAction::Enum SIGHUP_action):
  SIGINT_action_(SIGINT_action),
  SIGHUP_action_(SIGHUP_action) {
  HookupHandler();
}

SignalHandler::~SignalHandler() {
  UnhookHandler();
}

SolverAction::Enum SignalHandler::CheckForSignals() const {
  if (GotSignal()) {
    switch (received_signal) {
#ifdef SIGHUP
    case SIGHUP:
      return SIGHUP_action_;
#endif
    case SIGINT:
      return SIGINT_action_;
    }
  }
  return SolverAction::NONE;
}

// Return the function that the solver can use to find out if a snapshot or
// early exit is being requested.
ActionCallback SignalHandler::GetActionFunction() {
  return boost::bind(&SignalHandler::CheckForSignals, this);
}

}  // namespace caffe
