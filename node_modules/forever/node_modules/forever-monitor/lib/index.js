/*
 * index.js: Top-level include for the `forever-monitor` module.
 *
 * (C) 2010 Charlie Robbins & the Contributors
 * MIT LICENCE
 *
 */

var utile = require('utile'),
    common = require('./forever-monitor/common');

exports.kill         = common.kill;
exports.checkProcess = common.checkProcess;
exports.Monitor      = require('./forever-monitor/monitor').Monitor;

//
// Expose version through `pkginfo`
//
exports.version = require('../package').version;

//
// ### function start (script, options)
// #### @script {string} Location of the script to run.
// #### @options {Object} Configuration for forever instance.
// Starts a script with forever
//
exports.start = function (script, options) {
  if (!options.uid) {
    options.uid = options.uid || utile.randomString(4).replace(/^\-/, '_');
  }

  return new exports.Monitor(script, options).start();
};
