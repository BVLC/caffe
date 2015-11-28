var isArrayLike = require('./isArrayLike'),
    isObject = require('../lang/isObject'),
    isString = require('../lang/isString'),
    support = require('../support'),
    values = require('../object/values');

/**
 * Converts `value` to an array-like object if it's not one.
 *
 * @private
 * @param {*} value The value to process.
 * @returns {Array|Object} Returns the array-like object.
 */
function toIterable(value) {
  if (value == null) {
    return [];
  }
  if (!isArrayLike(value)) {
    return values(value);
  }
  if (support.unindexedChars && isString(value)) {
    return value.split('');
  }
  return isObject(value) ? value : Object(value);
}

module.exports = toIterable;
