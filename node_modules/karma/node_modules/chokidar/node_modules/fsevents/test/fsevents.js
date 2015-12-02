/*
 ** © 2014 by Philipp Dunkel <pip@pipobscure.com>
 ** Licensed under MIT License.
 */

/* jshint node:true */
'use strict';

var test = require('tap').test;

test('checking main module', function(t) {
  var mod = load('../');
  t.ok( !! mod, 'loading module');
  t.ok('function' === typeof mod, 'module.exports is an ' + (typeof mod));
  t.ok('function' === typeof mod.FSEvents, 'module.exports.FSEvents is an ' + (typeof mod));
  t.ok('function' === typeof mod.getInfo, 'module.exports.getInfo is an ' + (typeof mod));
  t.ok('object' === typeof mod.Constants, 'module.exports.Constants is an ' + (typeof mod));
  t.end();
});

function load(f) {
  try {
    return require(f);
  } catch (e) {
    return false;
  }
}
