'use strict';

var factories = require('./factories');

console.log('[hasHit = 5]');
require('do-you-even-bench')(factories.byTest('hasHit5'));
