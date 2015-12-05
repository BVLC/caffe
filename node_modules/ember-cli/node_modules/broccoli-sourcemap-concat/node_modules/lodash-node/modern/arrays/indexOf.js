/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize modern exports="node" -o ./modern/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var baseIndexOf = require('../internals/baseIndexOf'),
    sortedIndex = require('./sortedIndex');

/* Native method shortcuts for methods with the same name as other `lodash` methods */
var nativeMax = Math.max;

/**
 * Gets the index at which the first occurrence of `value` is found using
 * strict equality for comparisons, i.e. `===`. If the array is already sorted
 * providing `true` for `fromIndex` will run a faster binary search.
 *
 * @static
 * @memberOf _
 * @category Arrays
 * @param {Array} array The array to search.
 * @param {*} value The value to search for.
 * @param {boolean|number} [fromIndex=0] The index to search from or `true`
 *  to perform a binary search on a sorted array.
 * @returns {number} Returns the index of the matched value or `-1`.
 * @example
 *
 * _.indexOf([1, 2, 3, 1, 2, 3], 2);
 * // => 1
 *
 * _.indexOf([1, 2, 3, 1, 2, 3], 2, 3);
 * // => 4
 *
 * _.indexOf([1, 1, 2, 2, 3, 3], 2, true);
 * // => 2
 */
function indexOf(array, value, fromIndex) {
  if (typeof fromIndex == 'number') {
    var length = array ? array.length : 0;
    fromIndex = (fromIndex < 0 ? nativeMax(0, length + fromIndex) : fromIndex || 0);
  } else if (fromIndex) {
    var index = sortedIndex(array, value);
    return array[index] === value ? index : -1;
  }
  return baseIndexOf(array, value, fromIndex);
}

module.exports = indexOf;
