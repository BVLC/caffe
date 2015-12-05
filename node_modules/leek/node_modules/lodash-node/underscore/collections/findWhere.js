/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize underscore exports="node" -o ./underscore/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var where = require('./where');

/**
 * Examines each element in a `collection`, returning the first that
 * has the given properties. When checking `properties`, this method
 * performs a deep comparison between values to determine if they are
 * equivalent to each other.
 *
 * @static
 * @memberOf _
 * @category Collections
 * @param {Array|Object|string} collection The collection to iterate over.
 * @param {Object} properties The object of property values to filter by.
 * @returns {*} Returns the found element, else `undefined`.
 * @example
 *
 * var food = [
 *   { 'name': 'apple',  'organic': false, 'type': 'fruit' },
 *   { 'name': 'banana', 'organic': true,  'type': 'fruit' },
 *   { 'name': 'beet',   'organic': false, 'type': 'vegetable' }
 * ];
 *
 * _.findWhere(food, { 'type': 'vegetable' });
 * // => { 'name': 'beet', 'organic': false, 'type': 'vegetable' }
 */
function findWhere(object, properties) {
  return where(object, properties, true);
}

module.exports = findWhere;
