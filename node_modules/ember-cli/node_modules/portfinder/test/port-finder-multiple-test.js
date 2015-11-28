/*
 * portfinder-test.js: Tests for the `portfinder` module.
 *
 * (C) 2011, Charlie Robbins
 *
 */

var vows = require('vows'),
    assert = require('assert'),
    async = require('async'),
    http = require('http'),
    portfinder = require('../lib/portfinder');

var servers = [];

function createServers (callback) {
  var base = 8000;

  async.whilst(
    function () { return base < 8005 },
    function (next) {
      var server = http.createServer(function () { });
      server.listen(base, next);
      base++;
      servers.push(server);
    }, callback);
}

vows.describe('portfinder').addBatch({
  "When using portfinder module": {
    "with 5 existing servers": {
      topic: function () {
        createServers(this.callback);
      },
      "the getPorts() method with an argument of 3": {
        topic: function () {
          portfinder.getPorts(3, this.callback);
        },
        "should respond with the first three available ports (8005, 8006, 8007)": function (err, ports) {
          assert.isTrue(!err);
          assert.deepEqual(ports, [8005, 8006, 8007]);
        }
      }
    }
  }
}).addBatch({
  "When using portfinder module": {
    "with no existing servers": {
      topic: function () {
        servers.forEach(function (server) {
          server.close();
        });

        return null;
      },
      "the getPorts() method with an argument of 3": {
        topic: function () {
          portfinder.getPorts(3, this.callback);
        },
        "should respond with the first three available ports (8000, 8001, 80072": function (err, ports) {
          assert.isTrue(!err);
          assert.deepEqual(ports, [8000, 8001, 8002]);
        }
      }
    }
  }
}).export(module);