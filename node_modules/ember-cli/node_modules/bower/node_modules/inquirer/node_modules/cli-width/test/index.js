'use strict';

var tty = require('tty');
var test = require('tape');
var lib = require('../');


test('uses process.stdout.getWindowSize', function (t) {
  // mock stdout.getWindowSize
  process.stdout.getWindowSize = function () {
    return [100];
  };

  t.equal(lib(), 100, 'equal to mocked, 100');
  t.end();
});

test('uses tty.getWindowSize', function (t) {
  process.stdout.getWindowSize = undefined;
  tty.getWindowSize = function () {
    return [3, 5];
  };

  t.equal(lib(), 5, 'equal to mocked, 5');
  t.end();
});

test('uses custom env var', function (t) {
  tty.getWindowSize = undefined;
  process.env.CLI_WIDTH = 30;

  t.equal(lib(), 30, 'equal to mocked, 30');

  delete process.env.CLI_WIDTH;
  t.end();
});

test('uses default if env var is not a number', function (t) {
  process.env.CLI_WIDTH = 'foo';

  t.equal(lib(), 0, 'default unset value, 0');

  delete process.env.CLI_WIDTH;
  t.end();
});

test('uses default', function (t) {
  tty.getWindowSize = undefined;

  t.equal(lib(), 0, 'default unset value, 0');
  t.end();
});

test('uses overridden default', function (t) {
  lib.defaultWidth = 10;

  t.equal(lib(), 10, 'user-set default value, 10');
  t.end();
});
