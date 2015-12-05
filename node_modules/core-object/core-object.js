'use strict';

var assign = require('lodash-node/modern/objects/assign');

function CoreObject(options) {
  assign(this, options);
}

module.exports = CoreObject;

CoreObject.prototype.constructor = CoreObject;

CoreObject.extend = function(options) {
  var constructor = this;
  function Class() {
    constructor.apply(this, arguments);
    if (this.init) {
      this.init(options);
    }
  }

  Class.__proto__ = CoreObject;

  Class.prototype = Object.create(constructor.prototype);
  assign(Class.prototype, options);
  Class.prototype.constructor = Class;
  Class.prototype._super = constructor.prototype;

  return Class;
};

