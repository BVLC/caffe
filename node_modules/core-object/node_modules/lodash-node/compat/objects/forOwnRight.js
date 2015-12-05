/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize exports="node" -o ./compat/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var baseCreateCallback = require('../internals/baseCreateCallback'),
    keys = require('./keys');

/**
 * This method is like `_.forOwn` except that it iterates over elements
 * of a `collection` in the opposite order.
 *
 * @static
 * @memberOf _
 * @category Objects
 * @param {Object} object The object to iterate over.
 * @param {Function} [callback=identity] The function called per iteration.
 * @param {*} [thisArg] The `this` binding of `callback`.
 * @returns {Object} Returns `object`.
 * @example
 *
 * _.forOwnRight({ '0': 'zero', '1': 'one', 'length': 2 }, function(num, key) {
 *   console.log(key);
 * });
 * // => logs 'length', '1', and '0' assuming `_.forOwn` logs '0', '1', and 'length'
 */
function forOwnRight(object, callback, thisArg) {
  var props = keys(object),
      length = props.length;

  callback = baseCreateCallback(callback, thisArg, 3);
  while (length--) {
    var key = props[length];
    if (callback(object[key], key, object) === false) {
      break;
    }
  }
  return object;
}

module.exports = forOwnRight;
