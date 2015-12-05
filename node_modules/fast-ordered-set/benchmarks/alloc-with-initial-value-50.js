'use strict';

var factories = require('./factories');

console.log('[alloc-with-initial-value.size = 50]');
require('do-you-even-bench')(factories.byTest('create50'));
