/*
 * mount-test.js: Tests for the core mount method.
 *
 * (C) 2011, Charlie Robbins, Paolo Fragomeni, & the Contributors.
 * MIT LICENSE
 *
 */

var assert = require('assert'),
    vows = require('vows'),
    director = require('../../../lib/director');

vows.describe('director/cli/path').addBatch({
  "An instance of director.cli.Router with routes": {
    topic: new director.cli.Router({
      'apps': function () {
        console.log('apps');
      },
      ' users': function () {
        console.log('users');
      }
    }),
    "should create the correct nested routing table": function (router) {
      assert.isObject(router.routes.apps);
      assert.isFunction(router.routes.apps.on);
      assert.isObject(router.routes.users);
      assert.isFunction(router.routes.users.on);
    }
  }
}).export(module);
