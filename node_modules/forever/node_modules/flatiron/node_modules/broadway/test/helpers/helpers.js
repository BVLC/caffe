/*
 * helpers.js: Test helpers for using broadway.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */
 
var events = require('eventemitter2'),
    broadway = require('../../lib/broadway');

var helpers = exports;

helpers.findApp = function () {
  return Array.prototype.slice.call(arguments).filter(function (arg) {
    return arg instanceof events.EventEmitter2;
  })[0];
};

helpers.mockApp = function () {
  var mock = new events.EventEmitter2({ delimiter: '::', wildcard: true });
  mock.options = {};
  return mock;
};