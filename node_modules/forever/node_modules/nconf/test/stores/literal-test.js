/*
 * literal-test.js: Tests for the nconf literal store.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */

var vows = require('vows'),
    assert = require('assert'),
    helpers = require('../helpers'),
    nconf = require('../../lib/nconf');

vows.describe('nconf/stores/literal').addBatch({
  "An instance of nconf.Literal": {
    topic: new nconf.Literal({
      foo: 'bar',
      one: 2
    }),
    "should have the correct methods defined": function (literal) {
      assert.equal(literal.type, 'literal');
      assert.isFunction(literal.get);
      assert.isFunction(literal.set);
      assert.isFunction(literal.merge);
      assert.isFunction(literal.loadSync);
    },
    "should have the correct values in the store": function (literal) {
      assert.equal(literal.store.foo, 'bar');
      assert.equal(literal.store.one, 2);
    }
  }
}).export(module);