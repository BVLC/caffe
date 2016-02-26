#include <boost/bind.hpp>
#include <glog/logging.h>

#include <signal.h>
#include <csignal>

#include "caffe/util/signal_handler.h"

namespace {
  static volatile sig_atomic_t got_sigint = false;
  static volatile sig_atomic_t got_sighup = false;
  static bool already_hooked_up = false;

  void handle_signal(int signal) {
    switch (signal) {
#ifdef _MSC_VER
    case SIGBREAK:  // there is no SIGHUP in windows, take SIGBREAK instead.
      got_sighup = true;
      break;
#else
    case SIGHUP:
      got_sighup = true;
      break;
#endif
    case SIGINT:
      got_sigint = true;
      break;
    }
  }

  void HookupHandler() {
    if (already_hooked_up) {
      LOG(FATAL) << "Tried to hookup signal handlers more than once.";
    }
    already_hooked_up = true;
#ifdef _MSC_VER
    if (signal(SIGBREAK, handle_signal) == SIG_ERR) {
      LOG(FATAL) << "Cannot install SIGBREAK handler.";
    }
    if (signal(SIGINT, handle_signal) == SIG_ERR) {
      LOG(FATAL) << "Cannot install SIGINT handler.";
    }
#else
    struct sigaction sa;
    // Setup the handler
    sa.sa_handler = &handle_signal;
    // Restart the system call, if at all possible
    sa.sa_flags = SA_RESTART;
    // Block every signal during the handler
    sigfillset(&sa.sa_mask);
    // Intercept SIGHUP and SIGINT
    if (sigaction(SIGHUP, &sa, NULL) == -1) {
      LOG(FATAL) << "Cannot install SIGHUP handler.";
    }
    if (sigaction(SIGINT, &sa, NULL) == -1) {
      LOG(FATAL) << "Cannot install SIGINT handler.";
    }
#endif
  }

  // Set the signal handlers to the default.
  void UnhookHandler() {
    if (already_hooked_up) {
#ifdef _MSC_VER
      if (signal(SIGBREAK, SIG_DFL) == SIG_ERR) {
        LOG(FATAL) << "Cannot uninstall SIGBREAK handler.";
      }
      if (signal(SIGINT, SIG_DFL) == SIG_ERR) {
        LOG(FATAL) << "Cannot uninstall SIGINT handler.";
      }
#else
      struct sigaction sa;
      // Setup the sighub handler
      sa.sa_handler = SIG_DFL;
      // Restart the system call, if at all possible
      sa.sa_flags = SA_RESTART;
      // Block every signal during the handler
      sigfillset(&sa.sa_mask);
      // Intercept SIGHUP and SIGINT
      if (sigaction(SIGHUP, &sa, NULL) == -1) {
        LOG(FATAL) << "Cannot uninstall SIGHUP handler.";
      }
      if (sigaction(SIGINT, &sa, NULL) == -1) {
        LOG(FATAL) << "Cannot uninstall SIGINT handler.";
      }
#endif
      already_hooked_up = false;
    }
  }

  // Return true iff a SIGINT has been received since the last time this
  // function was called.
  bool GotSIGINT() {
    bool result = got_sigint;
    got_sigint = false;
    return result;
  }

  // Return true iff a SIGHUP has been received since the last time this
  // function was called.
  bool GotSIGHUP() {
    bool result = got_sighup;
    got_sighup = false;
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
  if (GotSIGHUP()) {
    return SIGHUP_action_;
  }
  if (GotSIGINT()) {
    return SIGINT_action_;
  }
  return SolverAction::NONE;
}

// Return the function that the solver can use to find out if a snapshot or
// early exit is being requested.
ActionCallback SignalHandler::GetActionFunction() {
  return boost::bind(&SignalHandler::CheckForSignals, this);
}

}  // namespace caffe
