/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize modern exports="node" -o ./modern/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */

module.exports = {
  'compact': require('./arrays/compact'),
  'difference': require('./arrays/difference'),
  'drop': require('./arrays/rest'),
  'findIndex': require('./arrays/findIndex'),
  'findLastIndex': require('./arrays/findLastIndex'),
  'first': require('./arrays/first'),
  'flatten': require('./arrays/flatten'),
  'head': require('./arrays/first'),
  'indexOf': require('./arrays/indexOf'),
  'initial': require('./arrays/initial'),
  'intersection': require('./arrays/intersection'),
  'last': require('./arrays/last'),
  'lastIndexOf': require('./arrays/lastIndexOf'),
  'object': require('./arrays/zipObject'),
  'pull': require('./arrays/pull'),
  'range': require('./arrays/range'),
  'remove': require('./arrays/remove'),
  'rest': require('./arrays/rest'),
  'sortedIndex': require('./arrays/sortedIndex'),
  'tail': require('./arrays/rest'),
  'take': require('./arrays/first'),
  'union': require('./arrays/union'),
  'uniq': require('./arrays/uniq'),
  'unique': require('./arrays/uniq'),
  'unzip': require('./arrays/zip'),
  'without': require('./arrays/without'),
  'xor': require('./arrays/xor'),
  'zip': require('./arrays/zip'),
  'zipObject': require('./arrays/zipObject')
};
