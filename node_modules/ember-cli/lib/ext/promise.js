'use strict';

var RSVP    = require('rsvp');
var Promise = RSVP.Promise;

module.exports = PromiseExt;

// Utility functions on the native CTOR need some massaging
module.exports.hash = function() {
  return this.resolve(RSVP.hash.apply(null, arguments));
};

module.exports.denodeify = function() {
  var fn = RSVP.denodeify.apply(null, arguments);
  var Constructor = this;
  var newFn = function() {
    return Constructor.resolve(fn.apply(null, arguments));
  };
  newFn.__proto__ = arguments[0];
  return newFn;
};

module.exports.filter = function() {
  return this.resolve(RSVP.filter.apply(null, arguments));
};

module.exports.map = function() {
  return this.resolve(RSVP.map.apply(null, arguments));
};

function PromiseExt(resolver, label) {
  this._superConstructor(resolver, label);
}

PromiseExt.prototype = Object.create(Promise.prototype);
PromiseExt.prototype.constructor = PromiseExt;
PromiseExt.prototype._superConstructor = Promise;
PromiseExt.__proto__ = Promise;

PromiseExt.prototype.returns = function(value) {
  return this.then(function() {
    return value;
  });
};

PromiseExt.prototype.invoke = function(method) {
  var args = Array.prototype.slice(arguments, 1);

  return this.then(function(value) {
    return value[method].apply(value, args);
  }, undefined, 'invoke: ' + method + ' with: ' + args);
};

function constructorMethod(promise, methodName, fn) {
  var Constructor = promise.constructor;

  return promise.then(function(values) {
    return Constructor[methodName](values, fn);
  });
}

PromiseExt.prototype.map = function(mapFn) {
  return constructorMethod(this, 'map', mapFn);
};

PromiseExt.prototype.filter = function(filterFn) {
  return constructorMethod(this, 'filter', filterFn);
};
