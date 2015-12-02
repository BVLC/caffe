/*
 * dispatch-test.js: Tests for the core dispatch method.
 *
 * (C) 2011, Charlie Robbins, Paolo Fragomeni, & the Contributors.
 * MIT LICENSE
 *
 */

var assert = require('assert'),
    vows = require('vows'),
    director = require('../../../lib/director');

vows.describe('director/cli/path').addBatch({
  "An instance of director.cli.Router": {
    topic: new director.cli.Router(),
    "the path() method": {
      "should create the correct nested routing table": function (router) {
        function noop () {}

        router.path(/apps/, function () {
          router.path(/foo/, function () {
            router.on(/bar/, noop);
          });

          router.on(/list/, noop);
        });

        router.on(/users/, noop);

        assert.isObject(router.routes.apps);
        assert.isFunction(router.routes.apps.list.on);
        assert.isObject(router.routes.apps.foo);
        assert.isFunction(router.routes.apps.foo.bar.on);
        assert.isFunction(router.routes.users.on);
      }
    }
  }
}).export(module);

