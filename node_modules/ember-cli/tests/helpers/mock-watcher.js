'use strict';

var RSVP         = require('rsvp');
var EventEmitter = require('events').EventEmitter;
var path         = require('path');

function MockWatcher() {
  EventEmitter.apply(this, arguments);
  this.tracks = [];
  this.trackTimings = [];
  this.trackErrors = [];
}

module.exports = MockWatcher;

MockWatcher.prototype = Object.create(EventEmitter.prototype);

MockWatcher.prototype.then = function() {
  var promise = RSVP.resolve({
    directory: path.resolve(__dirname, '../fixtures/express-server')
  });
  return promise.then.apply(promise, arguments);
};
