import {
  isArray,
  isMaybeThenable
} from './utils';

import {
  noop,
  reject,
  fulfill,
  subscribe,
  FULFILLED,
  REJECTED,
  PENDING
} from './-internal';

export function makeSettledResult(state, position, value) {
  if (state === FULFILLED) {
    return {
      state: 'fulfilled',
      value: value
    };
  } else {
     return {
      state: 'rejected',
      reason: value
    };
  }
}

function Enumerator(Constructor, input, abortOnReject, label) {
  var enumerator = this;

  enumerator._instanceConstructor = Constructor;
  enumerator.promise = new Constructor(noop, label);
  enumerator._abortOnReject = abortOnReject;

  if (enumerator._validateInput(input)) {
    enumerator._input     = input;
    enumerator.length     = input.length;
    enumerator._remaining = input.length;

    enumerator._init();

    if (enumerator.length === 0) {
      fulfill(enumerator.promise, enumerator._result);
    } else {
      enumerator.length = enumerator.length || 0;
      enumerator._enumerate();
      if (enumerator._remaining === 0) {
        fulfill(enumerator.promise, enumerator._result);
      }
    }
  } else {
    reject(enumerator.promise, enumerator._validationError());
  }
}

export default Enumerator;

Enumerator.prototype._validateInput = function(input) {
  return isArray(input);
};

Enumerator.prototype._validationError = function() {
  return new Error('Array Methods must be provided an Array');
};

Enumerator.prototype._init = function() {
  this._result = new Array(this.length);
};

Enumerator.prototype._enumerate = function() {
  var enumerator = this;
  var length     = enumerator.length;
  var promise    = enumerator.promise;
  var input      = enumerator._input;

  for (var i = 0; promise._state === PENDING && i < length; i++) {
    enumerator._eachEntry(input[i], i);
  }
};

Enumerator.prototype._eachEntry = function(entry, i) {
  var enumerator = this;
  var c = enumerator._instanceConstructor;
  if (isMaybeThenable(entry)) {
    if (entry.constructor === c && entry._state !== PENDING) {
      entry._onError = null;
      enumerator._settledAt(entry._state, i, entry._result);
    } else {
      enumerator._willSettleAt(c.resolve(entry), i);
    }
  } else {
    enumerator._remaining--;
    enumerator._result[i] = enumerator._makeResult(FULFILLED, i, entry);
  }
};

Enumerator.prototype._settledAt = function(state, i, value) {
  var enumerator = this;
  var promise = enumerator.promise;

  if (promise._state === PENDING) {
    enumerator._remaining--;

    if (enumerator._abortOnReject && state === REJECTED) {
      reject(promise, value);
    } else {
      enumerator._result[i] = enumerator._makeResult(state, i, value);
    }
  }

  if (enumerator._remaining === 0) {
    fulfill(promise, enumerator._result);
  }
};

Enumerator.prototype._makeResult = function(state, i, value) {
  return value;
};

Enumerator.prototype._willSettleAt = function(promise, i) {
  var enumerator = this;

  subscribe(promise, undefined, function(value) {
    enumerator._settledAt(FULFILLED, i, value);
  }, function(reason) {
    enumerator._settledAt(REJECTED, i, reason);
  });
};
