/*
 * stopall-peaceful-test.js: tests if `forever start` followed by `forever stopall` works.
 *
 * (C) 2010 Charlie Robbins & the Contributors
 * MIT LICENCE
 *
 */

var assert = require('assert'),
  path = require('path'),
  fs = require('fs'),
  spawn = require('child_process').spawn,
  vows = require('vows'),
  forever = require('../../lib/forever');

function runCmd(cmd, args) {
  var proc = spawn(process.execPath, [
    path.resolve(__dirname, '../../', 'bin/forever'),
    cmd
  ].concat(args), {detached: true});
  proc.unref();
  return proc;
}

vows.describe('forever/core/stopall-peaceful').addBatch({
  "When using forever" : {
    "to run script with 100% exit" : {
      topic: function () {
        runCmd('start', [
          './test/fixtures/cluster-fork-mode.js'
        ]);
        setTimeout(function (that) {
          forever.list(false, that.callback);
        }, 2000, this);
      },
      "the script should be marked as `STOPPED`": function (err, procs) {
        assert.isNull(err);
        assert.isArray(procs);
        assert.equal(procs.length, 1);
        assert.ok(!procs[0].running);
      }
    }
  }
}).addBatch({
  "When the script is running" : {
    "try to stop all" : {
      topic: function () {
        var that = this;
        forever.list(false, function(err, procs) {
          assert.isNull(err);
          assert.isArray(procs);
          assert.equal(procs.length, 1);
          // get pid.
          var pid = procs[0].pid;
          // run command
          var cmd = runCmd('stopall', []);
          cmd.stdout.on('data', onData);
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