/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize modern exports="node" -o ./modern/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var createCallback = require('../functions/createCallback'),
    identity = require('../utilities/identity');

/**
 * Uses a binary search to determine the smallest index at which a value
 * should be inserted into a given sorted array in order to maintain the sort
 * order of the array. If a callback is provided it will be executed for
 * `value` and each element of `array` to compute their sort ranking. The
 * callback is bound to `thisArg` and invoked with one argument; (value).
 *
 * If a property name is provided for `callback` the created "_.pluck" style
 * callback will return the property value of the given element.
 *
 * If an object is provided for `callback` the created "_.where" style callback
 * will return `true` for elements that have the properties of the given object,
 * else `false`.
 *
 * @static
 * @memberOf _
 * @category Arrays
 * @param {Array} array The array to inspect.
 * @param {*} value The value to evaluate.
 * @param {Function|Object|string} [callback=identity] The function called
 *  per iteration. If a property name or object is provided it will be used
 *  to create a "_.pluck" or "_.where" style callback, respectively.
 * @param {*} [thisArg] The `this` binding of `callback`.
 * @returns {number} Returns the index at which `value` should be inserted
 *  into `array`.
 * @example
 *
 * _.sortedIndex([20, 30, 50], 40);
 * // => 2
 *
 * // using "_.pluck" callback shorthand
 * _.sortedIndex([{ 'x': 20 }, { 'x': 30 }, { 'x': 50 }], { 'x': 40 }, 'x');
 * // => 2
 *
 * var dict = {
 *   'wordToNumber': { 'twenty': 20, 'thirty': 30, 'fourty': 40, 'fifty': 50 }
 * };
 *
 * _.sortedIndex(['twenty', 'thirty', 'fifty'], 'fourty', function(word) {
 *   return dict.wordToNumber[word];
 * });
 * // => 2
 *
 * _.sortedIndex(['twenty', 'thirty', 'fifty'], 'fourty', function(word) {
 *   return this.wordToNumber[word];
 * }, dict);
 * // => 2
 */
function sortedIndex(array, value, callback, thisArg) {
  var low = 0,
      high = array ? array.length : low;

  // explicitly reference `identity` for better inlining in Firefox
  callback = callback ? createCallback(callback, thisArg, 1) : identity;
  value = callback(value);

  while (low < high) {
    var mid = (low + high) >>> 1;
    (callback(array[mid]) < value)
      ? low = mid + 1
      : high = mid;
  }
  return low;
}

module.exports = sortedIndex;
