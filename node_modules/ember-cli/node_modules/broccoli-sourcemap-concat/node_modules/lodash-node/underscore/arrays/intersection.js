/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize underscore exports="node" -o ./underscore/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var baseIndexOf = require('../internals/baseIndexOf'),
    isArguments = require('../objects/isArguments'),
    isArray = require('../objects/isArray');

/**
 * Creates an array of unique values present in all provided arrays using
 * strict equality for comparisons, i.e. `===`.
 *
 * @static
 * @memberOf _
 * @category Arrays
 * @param {...Array} [array] The arrays to inspect.
 * @returns {Array} Returns an array of shared values.
 * @example
 *
 * _.intersection([1, 2, 3], [5, 2, 1, 4], [2, 1]);
 * // => [1, 2]
 */
function intersection() {
  var args = [],
      argsIndex = -1,
      argsLength = arguments.length;

  while (++argsIndex < argsLength) {
    var value = arguments[argsIndex];
     if (isArray(value) || isArguments(value)) {
       args.push(value);
     }
  }
  var array = args[0],
      index = -1,
      indexOf = baseIndexOf,
      length = array ? array.length : 0,
      result = [];

  outer:
  while (++index < length) {
    value = array[index];
    if (indexOf(result, value) < 0) {
      var argsIndex = argsLength;
      while (--argsIndex) {
        if (indexOf(args[argsIndex], value) < 0) {
          continue outer;
        }
      }
      result.push(value);
    }
  }
  return result;
}

module.exports = intersection;
