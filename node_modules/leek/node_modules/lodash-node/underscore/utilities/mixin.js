/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize underscore exports="node" -o ./underscore/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var forEach = require('../collections/forEach'),
    functions = require('../objects/functions'),
    isFunction = require('../objects/isFunction');

/**
 * Used for `Array` method references.
 *
 * Normally `Array.prototype` would suffice, however, using an array literal
 * avoids issues in Narwhal.
 */
var arrayRef = [];

/** Native method shortcuts */
var push = arrayRef.push;

/**
 * Adds function properties of a source object to the destination object.
 * If `object` is a function methods will be added to its prototype as well.
 *
 * @static
 * @memberOf _
 * @category Utilities
 * @param {Function|Object} [object=lodash] object The destination object.
 * @param {Object} source The object of functions to add.
 * @param {Object} [options] The options object.
 * @param {boolean} [options.chain=true] Specify whether the functions added are chainable.
 * @example
 *
 * function capitalize(string) {
 *   return string.charAt(0).toUpperCase() + string.slice(1).toLowerCase();
 * }
 *
 * _.mixin({ 'capitalize': capitalize });
 * _.capitalize('fred');
 * // => 'Fred'
 *
 * _('fred').capitalize().value();
 * // => 'Fred'
 *
 * _.mixin({ 'capitalize': capitalize }, { 'chain': false });
 * _('fred').capitalize();
 * // => 'Fred'
 */
function mixin(object, source) {
  var ctor = object,
      isFunc = isFunction(ctor);

  forEach(functions(source), function(methodName) {
    var func = object[methodName] = source[methodName];
    if (isFunc) {
      ctor.prototype[methodName] = function() {
        var args = [this.__wrapped__];
        push.apply(args, arguments);

        var result = func.apply(object, args);
        if (this.__chain__) {
          result = new ctor(result);
          result.__chain__ = true;
        }
        return result;
      };
    }
  });
}

module.exports = mixin;
