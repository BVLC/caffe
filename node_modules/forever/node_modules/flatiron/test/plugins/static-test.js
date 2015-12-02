/*
 * static-test.js: Tests for flatiron app(s) using the static plugin
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */

var assert = require('assert'),
    fs = require('fs'),
    path = require('path'),
    request = require('request'),
    vows = require('vows');

var appDir = path.join(__dirname, '..', '..', 'examples', 'static-app'),
    app = require(path.join(appDir, 'app'));

vows.describe('flatiron/plugins/static').addBatch({
  "A flatiron app using `flatiron.plugins.static": {
    topic: app,
    "should extend the app correctly": function (app) {
      assert.isString(app._staticDir);
      assert.isFunction(app.static);
      assert.isFunction(app.http.before[0]);
    },
    "when the application is running": {
      topic: function () {
        app.start(8080, this.callback)
      },
      "a GET to /headers": {
        topic: function () {
          request('http://localhost:8080/headers', this.callback);
        },
        "should respond with JSON headers": function (err, res, body) {
          assert.isNull(err);
          assert.equal(res.statusCode, 200);
          assert.isObject(JSON.parse(body));
        }
      },
      "a GET to /style.css": {
        topic: function () {
          request('http://localhost:8080/style.css', this.callback);
        },
        "should respond with style.css file": function (err, res, body) {
          assert.isNull(err);
          assert.equal(res.statusCode, 200);

          assert.equal(
            fs.readFileSync(path.join(appDir, 'app', 'assets', 'style.css'), 'utf8'),
            body
          );
        }
      }
    }
  }
}).export(module);
