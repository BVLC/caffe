/*
 * daemonic-inheritance-test.js: Tests for configuration inheritance of forever.startDaemon()
 *
 * (C) 2010 Charlie Robbins & the Contributors
 * MIT LICENCE
 *
 */

var assert = require('assert'),
    path = require('path'),
    fs = require('fs'),
    vows = require('vows'),
    forever = require('../../lib/forever');

//
// n.b. (indexzero): The default root is `~/.forever` so this
// actually is a valid, non-default test path.
//
var myRoot = path.resolve(process.env.HOME, '.forever_root');

vows.describe('forever/core/startDaemon').addBatch({
  "When using forever" : {
    "the startDaemon() method with customized configuration" : {
      topic: function () {
        if (!fs.existsSync(myRoot)) {
          fs.mkdirSync(myRoot);
        }

        forever.load({root:myRoot});

        forever.startDaemon(path.join(__dirname, '..', 'fixtures', 'log-on-interval.js'));
        setTimeout(function (that) {
          forever.list(false, that.callback);
        }, 2000, this);
      },
      "should respond with 1 process": function (err, procs) {
        assert.isNull(err);
        assert.isArray(procs);
        assert.equal(procs.length, 1);
      },
      "and logs/pids/socks are all piping into the customized root": function (err, procs) {
        assert.equal(procs[0].logFile.indexOf(myRoot), 0);
        assert.equal(procs[0].pidFile.indexOf(myRoot), 0);
        assert.equal(procs[0].socket.indexOf(myRoot), 0);
      }
    }
  }
}).addBatch({
  "When the tests are over" : {
    "stop all forever processes" : {
      topic: function () {
        forever.load({root:myRoot});
        forever.stopAll().on('stopAll', this.callback.bind(null, null));
      },
      "should stop the correct number of procs": function (err, procs) {
        assert.isArray(procs);
        assert.lengthOf(procs, 1);
      }
    }
  }
}).export(module);
