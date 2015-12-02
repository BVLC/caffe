/*
 * http-test.js: Tests for basic HTTP server(s).
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

function assertBark(uri) {
  return macros.assertGet(
    9090,
    uri,
    'hello from (bark)'
  );
}

vows.describe('director/http').addBatch({
  "An instance of director.http.Router": {
    "instantiated with a Routing table": {
      topic: new director.http.Router({
        '/hello': {
          get: handlers.respondWithId
        }
      }),
      "should have the correct routes defined": function (router) {
        assert.isObject(router.routes.hello);
        assert.isFunction(router.routes.hello.get);
      },
      "when passed to an http.Server instance": {
        topic: function (router) {
          router.get(/foo\/bar\/(\w+)/, handlers.respondWithId);
          router.get(/add\/:param\/bark/, handlers.respondWithId);
          router.get(/foo\/update\/(\w+)/, handlers.respondWithId);
          router.path(/bar\/bazz\//, function () {
            this.get(/(\w+)/, handlers.respondWithId);
          });
          router.get(/\/foo\/wild\/(.*)/, handlers.respondWithId);
          router.get(/(\/v2)?\/somepath/, handlers.respondWithId);

          helpers.createServer(router)
            .listen(9090, this.callback);
        },
        "a request to foo/bar/bark": assertBark('foo/bar/bark'),
        "a request to foo/update/bark": assertBark('foo/update/bark'),
        "a request to bar/bazz/bark": assertBark('bar/bazz/bark'),
        "a request to foo/bar/bark?test=test": assertBark('foo/bar/bark?test=test'),
        "a request to foo/wild/bark": assertBark('foo/wild/bark'),
        "a request to add/:param/bark": {
          topic: function () {
            request({ uri: 'http://localhost:9090/add/test/bark' }, this.callback);
          },
          'should respond and parse the params': function (err, res, body) {
            assert.isNull(err);
            assert.equal(res.statusCode, 200);
            assert.equal(body, 'hello from (test)');
          }
        },
        "an encoded request to add/:param/bark": {
          topic: function () {
            request({ uri: 'http://localhost:9090/add/something%40example.com/bark' }, this.callback);
          },
          'should respond and parse the params': function (err, res, body) {
            assert.isNull(err);
            assert.equal(res.statusCode, 200);
            assert.equal(body, 'hello from (something%40example.com)');
          }
        },
        "an encoded request to add/:param/bark with additional special characters": {
          topic: function () {
            request({ uri: 'http://localhost:9090/add/something%40(example).com/bark' }, this.callback);
          },
          'should respond and parse the params': function (err, res, body) {
            assert.isNull(err);
            assert.equal(res.statusCode, 200);
            assert.equal(body, 'hello from (something%40(example).com)');
          }
        },
        "a request to foo/%RT": macros.assert404(9090, 'foo/%RT'),
        "a request to /v2/somepath": macros.assertGet(
          9090,
          '/v2/somepath',
          'hello from (/v2)'
        )
      }
    }
  }
}).export(module);
