export function objectOrFunction(x) {
  return typeof x === 'function' || (typeof x === 'object' && x !== null);
}

export function isFunction(x) {
  return typeof x === 'function';
}

export function isMaybeThenable(x) {
  return typeof x === 'object' && x !== null;
}

var _isArray;
if (!Array.isArray) {
  _isArray = function (x) {
    return Object.prototype.toString.call(x) === '[object Array]';
  };
} else {
  _isArray = Array.isArray;
}

export var isArray = _isArray;

// Date.now is not available in browsers < IE9
// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/now#Compatibility
export var now = Date.now || function() { return new Date().getTime(); };

function F() { }

export var o_create = (Object.create || function (o) {
  if (arguments.length > 1) {
    throw new Error('Second argument not supported');
  }
  if (typeof o !== 'object') {
    throw new TypeError('Argument must be an object');
  }
  F.prototype = o;
  return new F();
});
