/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize exports="node" -o ./compat/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */
var keys = require('../objects/keys');

/** Reusable iterator options for `assign` and `defaults` */
var defaultsIteratorOptions = {
  'args': 'object, source, guard',
  'top':
    'var args = arguments,\n' +
    '    argsIndex = 0,\n' +
    "    argsLength = typeof guard == 'number' ? 2 : args.length;\n" +
    'while (++argsIndex < argsLength) {\n' +
    '  iterable = args[argsIndex];\n' +
    '  if (iterable && objectTypes[typeof iterable]) {',
  'keys': keys,
  'loop': "if (typeof result[index] == 'undefined') result[index] = iterable[index]",
  'bottom': '  }\n}'
};

module.exports = defaultsIteratorOptions;
