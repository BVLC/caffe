/*
 * check-process-test.js: Tests for forever.checkProcess(pid)
 *
 * (C) 2010 Charlie Robbins & the Contributors
 * MIT LICENCE
 *
 */

var assert = require('assert'),
    path = require('path'),
    vows = require('vows'),
    fmonitor = require('../../lib');

vows.describe('forever/core/check-process').addBatch({
  "When using forever": {
    "checking if process exists": {
      "if process exists": {
        topic: fmonitor.checkProcess(process.pid),
        "should return true": function (result) {
          assert.isTrue(result);
        }
      },
      "if process doesn't exist": {
        topic: fmonitor.checkProcess(255 * 255 * 255),
        //
        // This is insanely large value. On most systems there'll be no process
        // with such PID. Also, there's no multiplatform way to check for
        // PID limit.
        //
        "should return false": function (result) {
          assert.isFalse(result);
        }
      }
    }
  }
}).export(module);
