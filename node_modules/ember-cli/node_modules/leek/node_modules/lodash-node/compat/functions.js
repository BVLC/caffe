/**
 * Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 * Build: `lodash modularize exports="node" -o ./compat/`
 * Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 * Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 * Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 * Available under MIT license <http://lodash.com/license>
 */

module.exports = {
  'after': require('./functions/after'),
  'bind': require('./functions/bind'),
  'bindAll': require('./functions/bindAll'),
  'bindKey': require('./functions/bindKey'),
  'compose': require('./functions/compose'),
  'createCallback': require('./functions/createCallback'),
  'curry': require('./functions/curry'),
  'debounce': require('./functions/debounce'),
  'defer': require('./functions/defer'),
  'delay': require('./functions/delay'),
  'memoize': require('./functions/memoize'),
  'once': require('./functions/once'),
  'partial': require('./functions/partial'),
  'partialRight': require('./functions/partialRight'),
  'throttle': require('./functions/throttle'),
  'wrap': require('./functions/wrap')
};
