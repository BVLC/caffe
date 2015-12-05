/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize exports="node" -o ./compat/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var isString = require('../objects/isString');

/** Used to detect and test whitespace */
var whitespace = (
  // whitespace
  ' \t\x0B\f\xA0\ufeff' +

  // line terminators
  '\n\r\u2028\u2029' +

  // unicode category "Zs" space separators
  '\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000'
);

/** Used to match leading whitespace and zeros to be removed */
var reLeadingSpacesAndZeros = RegExp('^[' + whitespace + ']*0+(?=.$)');

/* Native method shortcuts for methods with the same name as other `lodash` methods */
var nativeParseInt = global.parseInt;

/**
 * Converts the given value into an integer of the specified radix.
 * If `radix` is `undefined` or `0` a `radix` of `10` is used unless the
 * `value` is a hexadecimal, in which case a `radix` of `16` is used.
 *
 * Note: This method avoids differences in native ES3 and ES5 `parseInt`
 * implementations. See http://es5.github.io/#E.
 *
 * @static
 * @memberOf _
 * @category Utilities
 * @param {string} value The value to parse.
 * @param {number} [radix] The radix used to interpret the value to parse.
 * @returns {number} Returns the new integer value.
 * @example
 *
 * _.parseInt('08');
 * // => 8
 */
var parseInt = nativeParseInt(whitespace + '08') == 8 ? nativeParseInt : function(value, radix) {
  // Firefox < 21 and Opera < 15 follow the ES3 specified implementation of `parseInt`
  return nativeParseInt(isString(value) ? value.replace(reLeadingSpacesAndZeros, '') : value, radix || 0);
};

module.exports = parseInt;
