/*
 * env-test.js: Tests for the nconf env store.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */

var vows = require('vows'),
    assert = require('assert'),
    helpers = require('../helpers'),
    nconf = require('../../lib/nconf');

vows.describe('nconf/stores/env').addBatch({
  "An instance of nconf.Env": {
    topic: new nconf.Env(),
    "should have the correct methods defined": function (env) {
      assert.isFunction(env.loadSync);
      assert.isFunction(env.loadEnv);
      assert.isArray(env.whitelist);
      assert.lengthOf(env.whitelist, 0);
      assert.equal(env.separator, '');
    }
  }
}).export(module);
