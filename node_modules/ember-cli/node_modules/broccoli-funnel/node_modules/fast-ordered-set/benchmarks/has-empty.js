'use strict';

var factories = require('./factories');

console.log('[has-empty]');
require('do-you-even-bench')(factories.byTest('hasEmpty'));
