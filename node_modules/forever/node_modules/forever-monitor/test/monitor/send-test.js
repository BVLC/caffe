/*
 * send-test.js: Tests sending child processes messages.
 *
 * (C) 2013 Charlie Robbins & the Contributors
 * MIT LICENCE
 *
 */

var assert = require('assert'),
    path = require('path'),
    vows = require('vows'),
    fmonitor = require('../../lib');

vows.describe('forever-monitor/monitor/send').addBatch({
  "When using forever-monitor": {
    "and spawning a script": {
      "the parent process can send the child a message": {
        topic: function () {
          var script = path.join(__dirname, '..', 'fixtures', 'send-pong.js'),
              child = new (fmonitor.Monitor)(script, { silent: false, minUptime: 2000, max: 1, fork: true });

          child.on('message', this.callback.bind(null, null));
          child.start();
          child.send({from: 'parent'});
        },
        "should reemit the message correctly": function (err, msg) {
          assert.isObject(msg);
          assert.deepEqual(msg, {message: { from: 'parent' }, pong: true} );
        }
      }
    }
  }
}).export(module);