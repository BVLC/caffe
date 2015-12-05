/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize underscore exports="node" -o ./underscore/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var forIn = require('../objects/forIn'),
    indicatorObject = require('./indicatorObject'),
    isFunction = require('../objects/isFunction'),
    objectTypes = require('./objectTypes');

/** `Object#toString` result shortcuts */
var arrayClass = '[object Array]',
    boolClass = '[object Boolean]',
    dateClass = '[object Date]',
    numberClass = '[object Number]',
    objectClass = '[object Object]',
    regexpClass = '[object RegExp]',
    stringClass = '[object String]';

/** Used for native method references */
var objectProto = Object.prototype;

/** Used to resolve the internal [[Class]] of values */
var toString = objectProto.toString;

/** Native method shortcuts */
var hasOwnProperty = objectProto.hasOwnProperty;

/**
 * The base implementation of `_.isEqual`, without support for `thisArg` binding,
 * that allows partial "_.where" style comparisons.
 *
 * @private
 * @param {*} a The value to compare.
 * @param {*} b The other value to compare.
 * @param {Function} [callback] The function to customize comparing values.
 * @param {Function} [isWhere=false] A flag to indicate performing partial comparisons.
 * @param {Array} [stackA=[]] Tracks traversed `a` objects.
 * @param {Array} [stackB=[]] Tracks traversed `b` objects.
 * @returns {boolean} Returns `true` if the values are equivalent, else `false`.
 */
function baseIsEqual(a, b, stackA, stackB) {
  if (a === b) {
    return a !== 0 || (1 / a == 1 / b);
  }
  var type = typeof a,
      otherType = typeof b;

  if (a === a &&
      !(a && objectTypes[type]) &&
      !(b && objectTypes[otherType])) {
    return false;
  }
  if (a == null || b == null) {
    return a === b;
  }
  var className = toString.call(a),
      otherClass = toString.call(b);

  if (className != otherClass) {
    return false;
  }
  switch (className) {
    case boolClass:
    case dateClass:
      return +a == +b;

    case numberClass:
      return a != +a
        ? b != +b
        : (a == 0 ? (1 / a == 1 / b) : a == +b);

    case regexpClass:
    case stringClass:
      return a == String(b);
  }
  var isArr = className == arrayClass;
  if (!isArr) {
    var aWrapped = hasOwnProperty.call(a, '__wrapped__'),
        bWrapped = hasOwnProperty.call(b, '__wrapped__');

    if (aWrapped || bWrapped) {
      return baseIsEqual(aWrapped ? a.__wrapped__ : a, bWrapped ? b.__wrapped__ : b, stackA, stackB);
    }
    if (className != objectClass) {
      return false;
    }
    var ctorA = a.constructor,
        ctorB = b.constructor;

    if (ctorA != ctorB &&
          !(isFunction(ctorA) && ctorA instanceof ctorA && isFunction(ctorB) && ctorB instanceof ctorB) &&
          ('constructor' in a && 'constructor' in b)
        ) {
      return false;
    }
  }
  stackA || (stackA = []);
  stackB || (stackB = []);

  var length = stackA.length;
  while (length--) {
    if (stackA[length] == a) {
      return stackB[length] == b;
    }
  }
  var result = true,
      size = 0;

  stackA.push(a);
  stackB.push(b);

  if (isArr) {
    size = b.length;
    result = size == a.length;

    if (result) {
      while (size--) {
        if (!(result = baseIsEqual(a[size], b[size], stackA, stackB))) {
          break;
        }
      }
    }
  }
  else {
    forIn(b, function(value, key, b) {
      if (hasOwnProperty.call(b, key)) {
        size++;
        return !(result = hasOwnProperty.call(a, key) && baseIsEqual(a[key], value, stackA, stackB)) && indicatorObject;
      }
    });

    if (result) {
      forIn(a, function(value, key, a) {
        if (hasOwnProperty.call(a, key)) {
          return !(result = --size > -1) && indicatorObject;
        }
      });
    }
  }
  stackA.pop();
  stackB.pop();
  return result;
}

module.exports = baseIsEqual;
