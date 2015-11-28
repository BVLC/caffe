/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize underscore exports="node" -o ./underscore/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var max = require('../collections/max'),
    pluck = require('../collections/pluck');

/**
 * Creates an array of grouped elements, the first of which contains the first
 * elements of the given arrays, the second of which contains the second
 * elements of the given arrays, and so on.
 *
 * @static
 * @memberOf _
 * @alias unzip
 * @category Arrays
 * @param {...Array} [array] Arrays to process.
 * @returns {Array} Returns a new array of grouped elements.
 * @example
 *
 * _.zip(['fred', 'barney'], [30, 40], [true, false]);
 * // => [['fred', 30, true], ['barney', 40, false]]
 */
function zip() {
  var index = -1,
      length = max(pluck(arguments, 'length')),
      result = Array(length < 0 ? 0 : length);

  while (++index < length) {
    result[index] = pluck(arguments, index);
  }
  return result;
}

module.exports = zip;
