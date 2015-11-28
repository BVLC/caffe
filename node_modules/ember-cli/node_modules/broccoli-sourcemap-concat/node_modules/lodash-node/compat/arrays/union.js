/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize exports="node" -o ./compat/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var baseFlatten = require('../internals/baseFlatten'),
    baseUniq = require('../internals/baseUniq');

/**
 * Creates an array of unique values, in order, of the provided arrays using
 * strict equality for comparisons, i.e. `===`.
 *
 * @static
 * @memberOf _
 * @category Arrays
 * @param {...Array} [array] The arrays to inspect.
 * @returns {Array} Returns an array of combined values.
 * @example
 *
 * _.union([1, 2, 3], [5, 2, 1, 4], [2, 1]);
 * // => [1, 2, 3, 5, 4]
 */
function union() {
  return baseUniq(baseFlatten(arguments, true, true));
}

module.exports = union;
