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
  this._instanceConstructor = Constructor;
  this.promise = new Constructor(noop, label);
  this._abortOnReject = abortOnReject;

  if (this._validateInput(input)) {
    this._input     = input;
    this.length     = input.length;
    this._remaining = input.length;

    this._init();

    if (this.length === 0) {
      fulfill(this.promise, this._result);
    } else {
      this.length = this.length || 0;
      this._enumerate();
      if (this._remaining === 0) {
        fulfill(this.promise, this._result);
      }
    }
  } else {
    reject(this.promise, this._validationError());
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
  var length  = this.length;
  var promise = this.promise;
  var input   = this._input;

  for (var i = 0; promise._state === PENDING && i < length; i++) {
    this._eachEntry(input[i], i);
  }
};

Enumerator.prototype._eachEntry = function(entry, i) {
  var c = this._instanceConstructor;
  if (isMaybeThenable(entry)) {
    if (entry.constructor === c && entry._state !== PENDING) {
      entry._onError = null;
      this._settledAt(entry._state, i, entry._result);
    } else {
      this._willSettleAt(c.resolve(entry), i);
    }
  } else {
    this._remaining--;
    this._result[i] = this._makeResult(FULFILLED, i, entry);
  }
};

Enumerator.prototype._settledAt = function(state, i, value) {
  var promise = this.promise;

  if (promise._state === PENDING) {
    this._remaining--;

    if (this._abortOnReject && state === REJECTED) {
      reject(promise, value);
    } else {
      this._result[i] = this._makeResult(state, i, value);
    }
  }

  if (this._remaining === 0) {
    fulfill(promise, this._result);
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
