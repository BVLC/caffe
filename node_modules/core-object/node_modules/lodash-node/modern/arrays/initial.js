/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize modern exports="node" -o ./modern/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var createCallback = require('../functions/createCallback'),
    slice = require('../internals/slice');

/* Native method shortcuts for methods with the same name as other `lodash` methods */
var nativeMax = Math.max,
    nativeMin = Math.min;

/**
 * Gets all but the last element or last `n` elements of an array. If a
 * callback is provided elements at the end of the array are excluded from
 * the result as long as the callback returns truey. The callback is bound
 * to `thisArg` and invoked with three arguments; (value, index, array).
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
 * @param {Array} array The array to query.
 * @param {Function|Object|number|string} [callback=1] The function called
 *  per element or the number of elements to exclude. If a property name or
 *  object is provided it will be used to create a "_.pluck" or "_.where"
 *  style callback, respectively.
 * @param {*} [thisArg] The `this` binding of `callback`.
 * @returns {Array} Returns a slice of `array`.
 * @example
 *
 * _.initial([1, 2, 3]);
 * // => [1, 2]
 *
 * _.initial([1, 2, 3], 2);
 * // => [1]
 *
 * _.initial([1, 2, 3], function(num) {
 *   return num > 1;
 * });
 * // => [1]
 *
 * var characters = [
 *   { 'name': 'barney',  'blocked': false, 'employer': 'slate' },
 *   { 'name': 'fred',    'blocked': true,  'employer': 'slate' },
 *   { 'name': 'pebbles', 'blocked': true,  'employer': 'na' }
 * ];
 *
 * // using "_.pluck" callback shorthand
 * _.initial(characters, 'blocked');
 * // => [{ 'name': 'barney',  'blocked': false, 'employer': 'slate' }]
 *
 * // using "_.where" callback shorthand
 * _.pluck(_.initial(characters, { 'employer': 'na' }), 'name');
 * // => ['barney', 'fred']
 */
function initial(array, callback, thisArg) {
  var n = 0,
      length = array ? array.length : 0;

  if (typeof callback != 'number' && callback != null) {
    var index = length;
    callback = createCallback(callback, thisArg, 3);
    while (index-- && callback(array[index], index, array)) {
      n++;
    }
  } else {
    n = (callback == null || thisArg) ? 1 : callback || n;
  }
  return slice(array, 0, nativeMin(nativeMax(0, length - n), length));
}

module.exports = initial;
