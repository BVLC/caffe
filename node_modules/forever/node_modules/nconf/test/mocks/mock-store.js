/*
 * mock-store.js: Mock store for ensuring certain operations are actually called.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */

var util = require('util'),
    events = require('events'),
    nconf = require('../../lib/nconf');

var Mock = nconf.Mock = function () {
  events.EventEmitter.call(this);
  this.type = 'mock';
};

// Inherit from Memory store.
util.inherits(Mock, events.EventEmitter);

//
// ### function save (value, callback)
// #### @value {Object} _Ignored_ Left here for consistency
// #### @callback {function} Continuation to respond to when complete.
// Waits `1000ms` and then calls the callback and emits the `save` event.
//
Mock.prototype.save = function (value, callback) {
  if (!callback && typeof value === 'function') {
    callback = value;
    value = null;
  }
  
  var self = this;
  
  setTimeout(function () {
    self.emit('save');
    callback();
  }, 1000);
};