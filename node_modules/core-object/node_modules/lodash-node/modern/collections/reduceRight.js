/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize modern exports="node" -o ./modern/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var createCallback = require('../functions/createCallback'),
    forEachRight = require('./forEachRight');

/**
 * This method is like `_.reduce` except that it iterates over elements
 * of a `collection` from right to left.
 *
 * @static
 * @memberOf _
 * @alias foldr
 * @category Collections
 * @param {Array|Object|string} collection The collection to iterate over.
 * @param {Function} [callback=identity] The function called per iteration.
 * @param {*} [accumulator] Initial value of the accumulator.
 * @param {*} [thisArg] The `this` binding of `callback`.
 * @returns {*} Returns the accumulated value.
 * @example
 *
 * var list = [[0, 1], [2, 3], [4, 5]];
 * var flat = _.reduceRight(list, function(a, b) { return a.concat(b); }, []);
 * // => [4, 5, 2, 3, 0, 1]
 */
function reduceRight(collection, callback, accumulator, thisArg) {
  var noaccum = arguments.length < 3;
  callback = createCallback(callback, thisArg, 4);
  forEachRight(collection, function(value, index, collection) {
    accumulator = noaccum
      ? (noaccum = false, value)
      : callback(accumulator, value, index, collection);
  });
  return accumulator;
}

module.exports = reduceRight;
