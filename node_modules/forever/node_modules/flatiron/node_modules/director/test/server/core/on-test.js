/*
 * on-test.js: Tests for the on/route method.
 *
 * (C) 2011, Charlie Robbins, Paolo Fragomeni, & the Contributors.
 * MIT LICENSE
 *
 */

var assert = require('assert'),
    vows = require('vows'),
    director = require('../../../lib/director');

vows.describe('director/core/insert').addBatch({
  "An instance of director.Router": {
    topic: new director.Router(),
    "the on() method": {
      "['foo', 'bar']": function (router) {
        function noop () { }

        router.on(['foo', 'bar'], noop);
        assert.strictEqual(router.routes.foo.on, noop);
        assert.strictEqual(router.routes.bar.on, noop);
      },
      "'baz'": function (router) {
        function noop () { }

        router.on('baz', noop);
        assert.strictEqual(router.routes.baz.on, noop);
      },
      "'after', 'baz'": function (router) {
        function noop () { }

        router.on('after', 'boo', noop);
        assert.strictEqual(router.routes.boo.after, noop);
      }
    }
  }
}).export(module);
