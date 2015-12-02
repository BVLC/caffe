/*
 * assert.js: Assertion helpers for broadway tests
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var assert = module.exports = require('assert'),
    fs = require('fs'),
    path = require('path'),
    nconf = require('nconf'),
    vows = require('vows');

//
// ### Assertion helpers for working with `broadway.App` objects.
//
assert.app = {};

//
// ### Assertion helpers for working with `broadway.plugins`.
//
assert.plugins = {};

//
// ### Assert that an application has various plugins.
//
assert.plugins.has = {
  config: function (app, config) {
    assert.instanceOf(app.config, nconf.Provider);
    if (config) {
      //
      // TODO: Assert that all configuration has been loaded
      //
    }
  },
  exceptions: function (app) {

  },
  directories: function (app) {
    if (app.options['directories']) {
      Object.keys(app.options['directories']).forEach(function (key) {
        assert.isTrue((fs.existsSync || path.existsSync)(app.options['directories'][key]));
      });
    }
  },
  log: function (app) {
    assert.isObject(app.log);

    //
    // TODO: Assert winston.extend methods
    //
  }
};

//
// ### Assert that an application doesn't have various plugins
//
assert.plugins.notHas = {
  config: function (app) {
    assert.isTrue(!app.config);
  },
  exceptions: function (app) {

  },
  directories: function (app) {
    assert.isTrue(!app.config.get('directories'))
  },
  log: function (app) {
    assert.isTrue(!app.log);
    //
    // TODO: Assert winston.extend methods
    //
  }
};

assert.log = {};

assert.log.levelMsgMeta = function (err, level, msg, meta) {
  assert.equal(level, this.event[1]);
  assert.equal(msg, this.event[2]);
  assert.equal(meta, this.event[3]);
};

assert.log.msgMeta = function (err, level, msg, meta) {
  assert.equal(level, this.event[0].split('::')[1] || 'info');
  assert.equal(msg, this.event[1]);
  assert.equal(meta, this.event[2]);
};

assert.log.levelMeta = function (err, level, msg, meta) {
  assert.equal(level, this.event[1]);
  assert.equal(msg, this.event[0]);
  assert.deepEqual(meta, this.event[2]);
};

assert.log.levelMsg = function (err, level, msg, meta) {
  assert.equal(level, this.event[1]);
  assert.equal(msg, this.event[2]);
};

assert.log.metaOnly = function (err, level, msg, meta, event) {
  assert.equal(level, 'info');
  assert.equal(msg, this.event[0]);
  assert.equal(meta, this.event[1]);
  assert.equal(event, this.event[0]);
};
