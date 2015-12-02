'use strict';

var utils = require('lazy-cache')(require);
var fn = require;
require = utils;

require('expand-range', 'expand');
require('repeat-element', 'repeat');
require('preserve', 'tokens');

require = fn;
module.exports = utils;
