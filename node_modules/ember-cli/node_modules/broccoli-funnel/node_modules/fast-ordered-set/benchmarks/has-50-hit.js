'use strict';

var factories = require('./factories');

console.log('[hasHit = 50]');
require('do-you-even-bench')(factories.byTest('hasHit50'));
