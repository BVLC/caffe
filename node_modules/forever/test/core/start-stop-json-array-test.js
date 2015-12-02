/*
 * start-stop-json-test.js: start or stop forever using relative paths, the script path could be start with './', '../' ...
 *
 * (C) 2010 Charlie Robbins & the Contributors
 * MIT LICENCE
 *
 */

var assert = require('assert'),
    path = require('path'),
    fs = require('fs'),
    vows = require('vows'),
    async = require('utile').async,
    request = require('request'),
    forever = require('../../lib/forever'),
    runCmd = require('../helpers').runCmd;

vows.describe('forever/core/start-stop-json-array').addBatch({
  "When using forever" : {
    "to start process using JSON configuration file containing an array" : {
      topic: function () {
        runCmd('start', [
          './test/fixtures/servers.json'
        ]);
        setTimeout(function (that) {
          forever.list(false, that.callback);
        }, 2000, this);
      },
      "the startup should works fine": function (err, procs) {
        assert.isNull(err);
        assert.isArray(procs);
        assert.equal(procs.length, 2);
      }
    }
  }
}).addBatch({
  "When the script is running": {
    "request to both ports": {
      topic: function () {
        async.parallel({
          one: async.apply(request, { uri: 'http://localhost:8080', json: true }),
          two: async.apply(request, { uri: 'http://localhost:8081', json: true })
        }, this.callback);
      },
      "should respond correctly": function (err, results) {
        assert.isNull(err);
        assert.isTrue(!results.one[1].p);
        assert.equal(results.two[1].p, 8081);
      }
    }
  }
}).addBatch({
    "When the script is running" : {
      "try to stopall" : {
        topic: function () {
          runCmd('stopall', []);
          setTimeout(function (that) {
            forever.list(false, that.callback);
          }, 2000, this);
        },
        "the shut down should works fine": function (err, procs) {
          assert.isNull(err);
          assert.isNull(procs);
        }
      }
    }
  }).export(module);
