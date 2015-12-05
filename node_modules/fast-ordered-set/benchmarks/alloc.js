'use strict';

var factories = require('./factories');

console.log('[alloc]');
require('do-you-even-bench')(factories.byTest('create'));
