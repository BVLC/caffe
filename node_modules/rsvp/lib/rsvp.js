import Promise from './rsvp/promise';
import EventTarget from './rsvp/events';
import denodeify from './rsvp/node';
import all from './rsvp/all';
import allSettled from './rsvp/all-settled';
import race from './rsvp/race';
import hash from './rsvp/hash';
import hashSettled from './rsvp/hash-settled';
import rethrow from './rsvp/rethrow';
import defer from './rsvp/defer';
import {
  config,
  configure
} from './rsvp/config';
import map from './rsvp/map';
import resolve from './rsvp/resolve';
import reject from './rsvp/reject';
import filter from './rsvp/filter';
import asap from './rsvp/asap';

// defaults
config.async = asap;
config.after = function(cb) {
  setTimeout(cb, 0);
};
var cast = resolve;
function async(callback, arg) {
  config.async(callback, arg);
}

function on() {
  config['on'].apply(config, arguments);
}

function off() {
  config['off'].apply(config, arguments);
}

// Set up instrumentation through `window.__PROMISE_INTRUMENTATION__`
if (typeof window !== 'undefined' && typeof window['__PROMISE_INSTRUMENTATION__'] === 'object') {
  var callbacks = window['__PROMISE_INSTRUMENTATION__'];
  configure('instrument', true);
  for (var eventName in callbacks) {
    if (callbacks.hasOwnProperty(eventName)) {
      on(eventName, callbacks[eventName]);
    }
  }
}

export {
  cast,
  Promise,
  EventTarget,
  all,
  allSettled,
  race,
  hash,
  hashSettled,
  rethrow,
  defer,
  denodeify,
  configure,
  on,
  off,
  resolve,
  reject,
  async,
  map,
  filter
};
