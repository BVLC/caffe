'use strict';

var SilentError = require('silent-error');
var deprecate   = require('../utilities/deprecate');

Object.defineProperty(module, 'exports', {
  get: function () {
    deprecate('`ember-cli/lib/errors/silent.js` is deprecated,'+
      ' use `silent-error` instead.', true);
    return SilentError;
  }
});
