/*
 * literal.js: Simple literal Object store for nconf.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */

var util = require('util'),
    Memory = require('./memory').Memory

var Literal = exports.Literal = function Literal (options) {
  Memory.call(this, options);

  options       = options || {}
  this.type     = 'literal';
  this.readOnly = true;
  this.store    = options.store || options;
};

// Inherit from Memory store.
util.inherits(Literal, Memory);

//
// ### function loadSync (callback)
// Returns the data stored in `this.store` synchronously.
//
Literal.prototype.loadSync = function () {
  return this.store;
};