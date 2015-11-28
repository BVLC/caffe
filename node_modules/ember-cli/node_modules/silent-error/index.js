'use strict';
var debug = require('debug');

function SilentError(message) {
  if (!(this instanceof SilentError)) {
    throw new TypeError('SilentError must be instantiated with `new`');
  }

  this.name          = 'SilentError';
  this.message       = message;
  this.isSilentError = true;

  if (process.env.EMBER_VERBOSE_ERRORS === 'true') {
    this.stack = (new Error()).stack;
    this.suppressedStacktrace = false;
  } else {
    this.suppressedStacktrace = true;
  }
}

SilentError.prototype = Object.create(Error.prototype);
SilentError.prototype.constructor = SilentError;

SilentError.debugOrThrow = function debugOrThrow(label, e) {
  // if the error is a SilentError, ignore
  if(e && e.isSilentError) {
    // ignore this error, invalid blueprints are handled in run
    debug(label)(e);
  } else {
    // rethrow all other errors
    throw e;
  }
};

module.exports = SilentError;
