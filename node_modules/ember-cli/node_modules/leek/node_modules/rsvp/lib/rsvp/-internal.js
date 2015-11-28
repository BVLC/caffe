import {
  objectOrFunction,
  isFunction
} from './utils';

import instrument from './instrument';

import { config } from './config';

function  withOwnPromise() {
  return new TypeError('A promises callback cannot return that same promise.');
}

function noop() {}

var PENDING   = void 0;
var FULFILLED = 1;
var REJECTED  = 2;

var GET_THEN_ERROR = new ErrorObject();

function getThen(promise) {
  try {
    return promise.then;
  } catch(error) {
    GET_THEN_ERROR.error = error;
    return GET_THEN_ERROR;
  }
}

function tryThen(then, value, fulfillmentHandler, rejectionHandler) {
  try {
    then.call(value, fulfillmentHandler, rejectionHandler);
  } catch(e) {
    return e;
  }
}

function handleForeignThenable(promise, thenable, then) {
  config.async(function(promise) {
    var sealed = false;
    var error = tryThen(then, thenable, function(value) {
      if (sealed) { return; }
      sealed = true;
      if (thenable !== value) {
        resolve(promise, value);
      } else {
        fulfill(promise, value);
      }
    }, function(reason) {
      if (sealed) { return; }
      sealed = true;

      reject(promise, reason);
    }, 'Settle: ' + (promise._label || ' unknown promise'));

    if (!sealed && error) {
      sealed = true;
      reject(promise, error);
    }
  }, promise);
}

function handleOwnThenable(promise, thenable) {
  if (thenable._state === FULFILLED) {
    fulfill(promise, thenable._result);
  } else if (thenable._state === REJECTED) {
    thenable._onError = null;
    reject(promise, thenable._result);
  } else {
    subscribe(thenable, undefined, function(value) {
      if (thenable !== value) {
        resolve(promise, value);
      } else {
        fulfill(promise, value);
      }
    }, function(reason) {
      reject(promise, reason);
    });
  }
}

function handleMaybeThenable(promise, maybeThenable) {
  if (maybeThenable.constructor === promise.constructor) {
    handleOwnThenable(promise, maybeThenable);
  } else {
    var then = getThen(maybeThenable);

    if (then === GET_THEN_ERROR) {
      reject(promise, GET_THEN_ERROR.error);
    } else if (then === undefined) {
      fulfill(promise, maybeThenable);
    } else if (isFunction(then)) {
      handleForeignThenable(promise, maybeThenable, then);
    } else {
      fulfill(promise, maybeThenable);
    }
  }
}

function resolve(promise, value) {
  if (promise === value) {
    fulfill(promise, value);
  } else if (objectOrFunction(value)) {
    handleMaybeThenable(promise, value);
  } else {
    fulfill(promise, value);
  }
}

function publishRejection(promise) {
  if (promise._onError) {
    promise._onError(promise._result);
  }

  publish(promise);
}

function fulfill(promise, value) {
  if (promise._state !== PENDING) { return; }

  promise._result = value;
  promise._state = FULFILLED;

  if (promise._subscribers.length === 0) {
    if (config.instrument) {
      instrument('fulfilled', promise);
    }
  } else {
    config.async(publish, promise);
  }
}

function reject(promise, reason) {
  if (promise._state !== PENDING) { return; }
  promise._state = REJECTED;
  promise._result = reason;
  config.async(publishRejection, promise);
}

function subscribe(parent, child, onFulfillment, onRejection) {
  var subscribers = parent._subscribers;
  var length = subscribers.length;

  parent._onError = null;

  subscribers[length] = child;
  subscribers[length + FULFILLED] = onFulfillment;
  subscribers[length + REJECTED]  = onRejection;

  if (length === 0 && parent._state) {
    config.async(publish, parent);
  }
}

function publish(promise) {
  var subscribers = promise._subscribers;
  var settled = promise._state;

  if (config.instrument) {
    instrument(settled === FULFILLED ? 'fulfilled' : 'rejected', promise);
  }

  if (subscribers.length === 0) { return; }

  var child, callback, detail = promise._result;

  for (var i = 0; i < subscribers.length; i += 3) {
    child = subscribers[i];
    callback = subscribers[i + settled];

    if (child) {
      invokeCallback(settled, child, callback, detail);
    } else {
      callback(detail);
    }
  }

  promise._subscribers.length = 0;
}

function ErrorObject() {
  this.error = null;
}

var TRY_CATCH_ERROR = new ErrorObject();

function tryCatch(callback, detail) {
  try {
    return callback(detail);
  } catch(e) {
    TRY_CATCH_ERROR.error = e;
    return TRY_CATCH_ERROR;
  }
}

function invokeCallback(settled, promise, callback, detail) {
  var hasCallback = isFunction(callback),
      value, error, succeeded, failed;

  if (hasCallback) {
    value = tryCatch(callback, detail);

    if (value === TRY_CATCH_ERROR) {
      failed = true;
      error = value.error;
      value = null;
    } else {
      succeeded = true;
    }

    if (promise === value) {
      reject(promise, withOwnPromise());
      return;
    }

  } else {
    value = detail;
    succeeded = true;
  }

  if (promise._state !== PENDING) {
    // noop
  } else if (hasCallback && succeeded) {
    resolve(promise, value);
  } else if (failed) {
    reject(promise, error);
  } else if (settled === FULFILLED) {
    fulfill(promise, value);
  } else if (settled === REJECTED) {
    reject(promise, value);
  }
}

function initializePromise(promise, resolver) {
  var resolved = false;
  try {
    resolver(function resolvePromise(value){
      if (resolved) { return; }
      resolved = true;
      resolve(promise, value);
    }, function rejectPromise(reason) {
      if (resolved) { return; }
      resolved = true;
      reject(promise, reason);
    });
  } catch(e) {
    reject(promise, e);
  }
}

export {
  noop,
  resolve,
  reject,
  fulfill,
  subscribe,
  publish,
  publishRejection,
  initializePromise,
  invokeCallback,
  FULFILLED,
  REJECTED,
  PENDING
};
