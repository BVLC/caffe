/*
 * common.js: Common methods used in `forever-monitor`.
 *
 * (C) 2010 Charlie Robbins & the Contributors
 * MIT LICENCE
 *
 */

var psTree = require('ps-tree'),
    spawn = require('child_process').spawn;

//
// ### function checkProcess (pid, callback)
// #### @pid {string} pid of the process to check
// #### @callback {function} Continuation to pass control backto.
// Utility function to check to see if a pid is running
//
exports.checkProcess = function (pid) {
  if (!pid) {
    return false;
  }

  try {
    //
    // Trying to kill non-existent process here raises a ESRCH - no such
    // process exception. Also, signal 0 doesn't do no harm to a process - it
    // only checks if sending a signal to a given process is possible.
    //
    process.kill(pid, 0);
    return true;
  }
  catch (err) {
    return false;
  }
};

exports.kill = function (pid, killTree, signal, callback) {
  signal   = signal   || 'SIGKILL';
  callback = callback || function () {};

  if (killTree && process.platform !== 'win32') {
    psTree(pid, function (err, children) {
      [pid].concat(
        children.map(function (p) {
          return p.PID;
        })
      ).forEach(function (tpid) {
        try { process.kill(tpid, signal) }
        catch (ex) { }
      });

      callback();
    });
  }
  else {
    try { process.kill(pid, signal) }
    catch (ex) { }
    callback();
  }
};