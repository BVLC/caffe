'use strict';

var RSVP         = require('rsvp');
var EventEmitter = require('events').EventEmitter;
var path         = require('path');

function MockExpressServer() {
  EventEmitter.apply(this, arguments);
  this.tracks = [];
  this.trackTimings = [];
  this.trackErrors = [];
}

module.exports = MockExpressServer;

MockExpressServer.prototype = Object.create(EventEmitter.prototype);

MockExpressServer.prototype.then = function() {
  var promise = RSVP.resolve({
    directory: path.resolve(__dirname, '../fixtures/express-server')
  });
  return promise.then.apply(promise, arguments);
};
