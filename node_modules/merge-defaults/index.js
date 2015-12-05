var _ = require('lodash');

/**
 * defaultsDeep
 *
 * Implement a deep version of `_.defaults`.
 *
 * This method is hopefully temporary, until lodash has something
 * similar that can be called in a single method.  For now, it's
 * worth it to use a temporary module for readability.
 * (i.e. I know what `_.defaults` means offhand- not true for `_.partialRight`)
 */

// In case the end user decided to do `_.defaults = require('merge-defaults')`,
// before doing anything else, let's make SURE we have a reference to the original
// `_.defaults()` method definition.
var origLodashDefaults = _.defaults;

// Corrected: see https://github.com/lodash/lodash/issues/540
module.exports = _.partialRight(_.merge, function recursiveDefaults (dest,src) {

  // Ensure dates and arrays are not recursively merged
  if (_.isArray(arguments[0]) || _.isDate(arguments[0])) {
    return arguments[0];
  }
  return _.merge(dest, src, recursiveDefaults);
});

//origLodashDefaults.apply(_, Array.prototype.slice.call(arguments));

// module.exports = _.partialRight(_.merge, _.defaults);

// module.exports = _.partialRight(_.merge, function deep(a, b) {
//   // Ensure dates and arrays are not recursively merged
//   if (_.isArray(a) || _.isDate(a)) {
//     return a;
//   }
//   else return _.merge(a, b, deep);
// });