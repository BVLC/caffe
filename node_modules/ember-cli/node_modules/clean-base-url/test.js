"use strict";
var assert = require('assert');
var cleanBaseURL = require('.');

assert(cleanBaseURL('') === '/');
assert(cleanBaseURL('/') === '/');
assert(cleanBaseURL('ember') === '/ember/');
assert(cleanBaseURL('/ember') === '/ember/');
assert(cleanBaseURL('ember/') === '/ember/');
assert(cleanBaseURL('/ember/') === '/ember/');
assert(cleanBaseURL('ember/hamsters') === '/ember/hamsters/');
assert(cleanBaseURL('/ember/hamsters/') === '/ember/hamsters/');
assert(cleanBaseURL('app://localhost') === 'app://localhost/');
assert(cleanBaseURL('app://localhost/') === 'app://localhost/');
