function EqualityError(message, actual, expected) {
  this.message = message;
  this.actual = actual;
  this.expected = expected;
  this.showDiff = true;
  Error.captureStackTrace(this, module.exports);
}

EqualityError.prototype = Object.create(Error.prototype);
EqualityError.prototype.name = 'EqualityError';
EqualityError.prototype.constructor = EqualityError;

var assert = require('assert');
var esprima = require('esprima');
var recast = require('recast');

module.exports = function(actual, expected, message) {
  var parsedActual   = esprima.parse(actual);
  var parsedExpected = esprima.parse(expected);

  var seemEqual = JSON.stringify(parsedActual) === JSON.stringify(parsedExpected);

  if (!seemEqual) {
    throw new EqualityError(message || "AST equality failed",
      recast.print(parsedActual).code,
      recast.print(parsedExpected).code
    );
  }
};
