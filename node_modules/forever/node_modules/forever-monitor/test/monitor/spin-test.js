/*
 * spin-test.js: Tests for spin restarts in forever.
 *
 * (C) 2010 Charlie Robbins & the Contributors
 * MIT LICENCE
 *
 */

var assert = require('assert'),
    path = require('path'),
    vows = require('vows'),
    fmonitor = require('../../lib');

vows.describe('forever-monitor/monitor/spin-restart').addBatch({
  "When using forever-monitor": {
    "and spawning a script that spin restarts": {
      "with no spinSleepTime specified": {
        topic: function () {
          var script = path.join(__dirname, '..', '..', 'examples', 'always-throw.js'),
              child = new (fmonitor.Monitor)(script, { silent: true, minUptime: 2000, max: 3 });

          child.on('exit', this.callback.bind({}, null));
          child.start();
        },
        "should spawn both processes appropriately": function (err, child, spinning) {
          assert.isTrue(spinning);
        },
        "should only run the bad process once": function (err, child, spinning) {
          assert.equal(child.times, 1);
        }
      },
      "with a spinSleepTime specified": {
        topic: function () {
          var script = path.join(__dirname, '..', '..', 'examples', 'always-throw.js'),
              child = new (fmonitor.Monitor)(script, { silent: true, max: 3, spinSleepTime: 1 });

          child.on('exit', this.callback.bind({}, null));
          child.start();
        },
        "should restart spinning processes": function (err, child, spinning) {
          assert.equal(child.times, 3);
        }
      }
    }
  }
}).export(module);
