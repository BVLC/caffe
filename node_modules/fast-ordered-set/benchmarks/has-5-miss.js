'use strict';

var factories = require('./factories');

console.log('[hasMiss = 5]');
require('do-you-even-bench')(factories.byTest('hasMiss5'));
