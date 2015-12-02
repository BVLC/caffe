/*
 * before-test.js: Tests for running before methods on HTTP server(s).
 *
 * (C) 2011, Charlie Robbins, Paolo Fragomeni, & the Contributors.
 * MIT LICENSE
 *
 */

var assert = require('assert'),
    http = require('http'),
    vows = require('vows'),
    request = require('request'),
    director = require('../../../lib/director'),
    helpers = require('../helpers'),
    handlers = helpers.handlers,
    macros = helpers.macros;

vows.describe('director/http/before').addBatch({
  "An instance of director.http.Router": {
    "with ad-hoc routes including .before()": {
      topic: function () {
        var router = new director.http.Router();

        router.before('/hello', function () { });
        router.after('/hello', function () { });
        router.get('/hello', handlers.respondWithId);

        return router;
      },
      "should have the correct routes defined": function (router) {
        assert.isObject(router.routes.hello);
        assert.isFunction(router.routes.hello.before);
        assert.isFunction(router.routes.hello.after);
        assert.isFunction(router.routes.hello.get);
      }
    }
  }
}).export(module);
