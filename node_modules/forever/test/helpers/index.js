/*
 * index.js: Test helpers for forever.
 *
 * (C) 2015 Charlie Robbins & the Contributors
 * MIT LICENCE
 *
 */

var path = require('path'),
    spawn = require('child_process').spawn;

/*
 * function runCmd (cmd, args)
 * Executes forever with the `cmd` and arguments.
 */
exports.runCmd = function runCmd(cmd, args) {
  var proc = spawn(process.execPath, [
    path.resolve(__dirname, '../../', 'bin/forever'),
    cmd
  ].concat(args), {detached: true});

  //
  // Pipe everything to `stderr` so it can
  // be seen when running `npm test`.
  //
  proc.stdout.pipe(process.stderr);
  proc.stderr.pipe(process.stderr);

  proc.unref();
  return proc;
}
