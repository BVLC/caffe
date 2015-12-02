/*
 * stopbypid-peaceful-test.js: tests if `forever start` followed by `forever stop <pid>` works.
 *
 * (C) 2010 Charlie Robbins & the Contributors
 * MIT LICENCE
 *
 */

var assert = require('assert'),
    path = require('path'),
    fs = require('fs'),
    vows = require('vows'),
    forever = require('../../lib/forever'),
    runCmd = require('../helpers').runCmd;

vows.describe('forever/core/stopbypid-peaceful').addBatch({
  "When using forever" : {
    "to run script with 100% exit" : {
      topic: function () {
        runCmd('start', [
          './test/fixtures/log-on-interval.js'
        ]);
        setTimeout(function (that) {
          forever.list(false, that.callback);
        }, 2000, this);
      },
      "the script should be running": function (err, procs) {
        assert.isNull(err);
        assert.isArray(procs);
        assert.equal(procs.length, 1);
        assert.ok(procs[0].running);
      }
    }
  }
}).addBatch({
  "When the script is running" : {
    "try to stop by pid" : {
      topic: function () {
        var that = this;
        forever.list(false, function(err, procs) {
          assert.isNull(err);
          assert.isArray(procs);
          assert.equal(procs.length, 1);
          // get pid.
          var pid = procs[0].pid;
          // run command
          var cmd = runCmd('stop', [pid]);
          cmd.stdout.on('data', onData);
          cmd.stderr.pipe(process.stderr);
          //listen on the `data` event.
          function onData(data) {
            // check whether pid exists or not.
            var line = data.toString().replace (/[\n\r\t\s]+/g, ' ');
            if (line && line.search(new RegExp(pid)) > 0) {
              that.callback(null, true);
              cmd.stdout.removeListener('data', onData);
            }
            // if pid did not exist, that means CLI has crashed, and no output was printed.
            // vows will raise an Asynchronous Error.
          }
        });
      },
      "the shut down should works fine": function (err, peaceful) {
        assert.isNull(err);
        assert.ok(peaceful);
      }
    }
  }
}).export(module);
