/*
 * stream-test.js: Tests for streaming HTTP in director.
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
    macros = helpers.macros,
    handlers = helpers.handlers

vows.describe('director/http/stream').addBatch({
  "An instance of director.http.Router": {
    "with streaming routes": {
      topic: function () {
        var router = new director.http.Router();
        router.post(/foo\/bar/, { stream: true }, handlers.streamBody);
        router.path('/a-path', function () {
          this.post({ stream: true }, handlers.streamBody);
        });

        return router;
      },
      "when passed to an http.Server instance": {
        topic: function (router) {
          helpers.createServer(router)
            .listen(9092, this.callback);
        },
        "a POST request to /foo/bar": macros.assertPost(9092, 'foo/bar', {
          foo: 'foo',
          bar: 'bar'
        }),
        "a POST request to /a-path": macros.assertPost(9092, 'a-path', {
          foo: 'foo',
          bar: 'bar'
        })
      }
    }
  }
}).export(module);
