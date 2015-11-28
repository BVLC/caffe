import Enumerator from './enumerator';
import {
  PENDING
} from './-internal';
import {
  o_create
} from './utils';

function PromiseHash(Constructor, object, label) {
  this._superConstructor(Constructor, object, true, label);
}

export default PromiseHash;

PromiseHash.prototype = o_create(Enumerator.prototype);
PromiseHash.prototype._superConstructor = Enumerator;
PromiseHash.prototype._init = function() {
  this._result = {};
};

PromiseHash.prototype._validateInput = function(input) {
  return input && typeof input === 'object';
};

PromiseHash.prototype._validationError = function() {
  return new Error('Promise.hash must be called with an object');
};

PromiseHash.prototype._enumerate = function() {
  var enumerator = this;
  var promise    = enumerator.promise;
  var input      = enumerator._input;
  var results    = [];

  for (var key in input) {
    if (promise._state === PENDING && Object.prototype.hasOwnProperty.call(input, key)) {
      results.push({
        position: key,
        entry: input[key]
      });
    }
  }

  var length = results.length;
  enumerator._remaining = length;
  var result;

  for (var i = 0; promise._state === PENDING && i < length; i++) {
    result = results[i];
    enumerator._eachEntry(result.entry, result.position);
  }
};
