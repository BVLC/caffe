/*
 * macros.js: Test macros for the forever-monitor module
 *
 * (C) 2010 Charlie Robbins & the Contributors
 * MIT LICENCE
 *
 */

var assert = require('assert'),
    path = require('path'),
    spawn = require('child_process').spawn,
    fmonitor = require('../../lib');

var macros = exports;

macros.assertTimes = function (script, times, options) {
  options.max = times;

  return {
    topic: function () {
      var child = new (fmonitor.Monitor)(script, options);
      child.on('exit', this.callback.bind({}, null));
      child.start();
    },
    "should emit 'exit' when completed": function (err, child) {
      assert.equal(child.times, times);
    }
  }
};

macros.assertStartsWith = function (string, substring) {
  assert.equal(string.slice(0, substring.length), substring);
};

macros.assertList = function (list) {
  assert.isNotNull(list);
  assert.lengthOf(list, 1);
};