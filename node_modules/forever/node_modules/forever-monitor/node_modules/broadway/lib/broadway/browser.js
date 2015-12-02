
/*
 * browser.js: Browser specific functionality for broadway.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var id = 0;

var common = {
  mixin: function (target) {
    var objs = Array.prototype.slice.call(arguments, 1);
    objs.forEach(function (o) {
      Object.keys(o).forEach(function (attr) {
        var getter = o.__lookupGetter__(attr);
        if (!getter) {
          target[attr] = o[attr];
        }
        else {
          target.__defineGetter__(attr, getter);
        }
      });
    });

    return target;
  },
  uuid: function () {
    return String(id++);
  }
};

var App = exports.App = function (options) {
  //
  // Setup options and `App` constants.
  //
  var self       = this;
  options        = options || {};
  this.root      = options.root;
  this.delimiter = options.delimiter || '::';

  //
  // Inherit from `EventEmitter2`
  //
  exports.EventEmitter2.call(this, {
    delimiter: this.delimiter,
    wildcard: true
  });

  //
  // Setup other relevant options such as the plugins
  // for this instance.
  //
  this.options      = options;
  this.plugins      = options.plugins || {};
  this.initialized  = false;
  this.bootstrapper = { init: function (app, func) {} };
  this.initializers = {};
};

var inherit = function (ctor, superCtor) {
  ctor.super_ = superCtor;
  ctor.prototype = Object.create(superCtor.prototype, {
    constructor: {
      value: ctor,
      enumerable: false,
      writable: true,
      configurable: true
    }
  });
}

inherit(exports.App, exports.EventEmitter2);

