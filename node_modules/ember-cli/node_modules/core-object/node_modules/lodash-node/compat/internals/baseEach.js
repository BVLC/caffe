/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize exports="node" -o ./compat/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var createIterator = require('./createIterator'),
    eachIteratorOptions = require('./eachIteratorOptions');

/**
 * A function compiled to iterate `arguments` objects, arrays, objects, and
 * strings consistenly across environments, executing the callback for each
 * element in the collection. The callback is bound to `thisArg` and invoked
 * with three arguments; (value, index|key, collection). Callbacks may exit
 * iteration early by explicitly returning `false`.
 *
 * @private
 * @type Function
 * @param {Array|Object|string} collection The collection to iterate over.
 * @param {Function} [callback=identity] The function called per iteration.
 * @param {*} [thisArg] The `this` binding of `callback`.
 * @returns {Array|Object|string} Returns `collection`.
 */
var baseEach = createIterator(eachIteratorOptions);

module.exports = baseEach;
