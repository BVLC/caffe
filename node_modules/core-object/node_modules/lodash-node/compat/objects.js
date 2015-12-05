/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize exports="node" -o ./compat/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */

module.exports = {
  'assign': require('./objects/assign'),
  'clone': require('./objects/clone'),
  'cloneDeep': require('./objects/cloneDeep'),
  'create': require('./objects/create'),
  'defaults': require('./objects/defaults'),
  'extend': require('./objects/assign'),
  'findKey': require('./objects/findKey'),
  'findLastKey': require('./objects/findLastKey'),
  'forIn': require('./objects/forIn'),
  'forInRight': require('./objects/forInRight'),
  'forOwn': require('./objects/forOwn'),
  'forOwnRight': require('./objects/forOwnRight'),
  'functions': require('./objects/functions'),
  'has': require('./objects/has'),
  'invert': require('./objects/invert'),
  'isArguments': require('./objects/isArguments'),
  'isArray': require('./objects/isArray'),
  'isBoolean': require('./objects/isBoolean'),
  'isDate': require('./objects/isDate'),
  'isElement': require('./objects/isElement'),
  'isEmpty': require('./objects/isEmpty'),
  'isEqual': require('./objects/isEqual'),
  'isFinite': require('./objects/isFinite'),
  'isFunction': require('./objects/isFunction'),
  'isNaN': require('./objects/isNaN'),
  'isNull': require('./objects/isNull'),
  'isNumber': require('./objects/isNumber'),
  'isObject': require('./objects/isObject'),
  'isPlainObject': require('./objects/isPlainObject'),
  'isRegExp': require('./objects/isRegExp'),
  'isString': require('./objects/isString'),
  'isUndefined': require('./objects/isUndefined'),
  'keys': require('./objects/keys'),
  'mapValues': require('./objects/mapValues'),
  'merge': require('./objects/merge'),
  'methods': require('./objects/functions'),
  'omit': require('./objects/omit'),
  'pairs': require('./objects/pairs'),
  'pick': require('./objects/pick'),
  'transform': require('./objects/transform'),
  'values': require('./objects/values')
};
