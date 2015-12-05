/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize exports="node" -o ./compat/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var createIterator = require('../internals/createIterator'),
    defaultsIteratorOptions = require('../internals/defaultsIteratorOptions');

/**
 * Assigns own enumerable properties of source object(s) to the destination
 * object. Subsequent sources will overwrite property assignments of previous
 * sources. If a callback is provided it will be executed to produce the
 * assigned values. The callback is bound to `thisArg` and invoked with two
 * arguments; (objectValue, sourceValue).
 *
 * @static
 * @memberOf _
 * @type Function
 * @alias extend
 * @category Objects
 * @param {Object} object The destination object.
 * @param {...Object} [source] The source objects.
 * @param {Function} [callback] The function to customize assigning values.
 * @param {*} [thisArg] The `this` binding of `callback`.
 * @returns {Object} Returns the destination object.
 * @example
 *
 * _.assign({ 'name': 'fred' }, { 'employer': 'slate' });
 * // => { 'name': 'fred', 'employer': 'slate' }
 *
 * var defaults = _.partialRight(_.assign, function(a, b) {
 *   return typeof a == 'undefined' ? b : a;
 * });
 *
 * var object = { 'name': 'barney' };
 * defaults(object, { 'name': 'fred', 'employer': 'slate' });
 * // => { 'name': 'barney', 'employer': 'slate' }
 */
var assign = createIterator(defaultsIteratorOptions, {
  'top':
    defaultsIteratorOptions.top.replace(';',
      ';\n' +
      "if (argsLength > 3 && typeof args[argsLength - 2] == 'function') {\n" +
      '  var callback = baseCreateCallback(args[--argsLength - 1], args[argsLength--], 2);\n' +
      "} else if (argsLength > 2 && typeof args[argsLength - 1] == 'function') {\n" +
      '  callback = args[--argsLength];\n' +
      '}'
    ),
  'loop': 'result[index] = callback ? callback(result[index], iterable[index]) : iterable[index]'
});

module.exports = assign;
