/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize modern exports="node" -o ./modern/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */

/** Native method shortcuts */
var floor = Math.floor;

/* Native method shortcuts for methods with the same name as other `lodash` methods */
var nativeRandom = Math.random;

/**
 * The base implementation of `_.random` without argument juggling or support
 * for returning floating-point numbers.
 *
 * @private
 * @param {number} min The minimum possible value.
 * @param {number} max The maximum possible value.
 * @returns {number} Returns a random number.
 */
function baseRandom(min, max) {
  return min + floor(nativeRandom() * (max - min + 1));
}

module.exports = baseRandom;
