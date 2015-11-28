'use strict';

var RSVP           = require('rsvp');
var Promise        = require('../../lib/ext/promise');
var chalk          = require('chalk');
var spawn          = require('child_process').spawn;
var defaults       = require('lodash/object/defaults');
var killCliProcess = require('./kill-cli-process');
var logOnFailure   = require('./log-on-failure');

module.exports = function run(/* command, args, options */) {
  var command = arguments[0];
  var args = Array.prototype.slice.call(arguments, 1);
  var options = {};

  if (typeof args[args.length - 1] === 'object') {
    options = args.pop();
  }

  options = defaults(options, {
    // If true, pass through stdout/stderr.
    // If false, only pass through stdout/stderr if the current test fails.
    verbose: false,

    onOutput: function(string) {
      options.log(string);
    },

    onError: function(string) {
      options.log(chalk.red(string));
    },

    log: function(string) {
      if (options.verbose) {
        console.log(string);
      } else {
        logOnFailure(string);
      }
    }
  });

  return new RSVP.Promise(function(resolve, reject) {
    options.log('      Running: ' + command + ' ' + args.join(' ') + ' in: ' + process.cwd());

    var opts = {};
    if (process.platform === 'win32') {
      args = ['"' + command + '"'].concat(args);
      command = 'node';
      opts.windowsVerbatimArguments = true;
      opts.stdio = [null, null, null, 'ipc'];
    }
    var child = spawn(command, args, opts);
    var result = {
      output: [],
      errors: [],
      code: null
    };

    if (options.onChildSpawned) {
      var onChildSpawnedPromise = new Promise(function (childSpawnedResolve, childSpawnedReject) {
        try {
          options.onChildSpawned(child).then(childSpawnedResolve, childSpawnedReject);
        } catch (err) {
          childSpawnedReject(err);
        }
      });
      onChildSpawnedPromise
        .then(function () {
          if (options.killAfterChildSpawnedPromiseResolution) {
            killCliProcess(child);
          }
        }, function (err) {
          result.testingError = err;
          if (options.killAfterChildSpawnedPromiseResolution) {
            killCliProcess(child);
          }
        });
    }

    child.stdout.on('data', function (data) {
      var string = data.toString();

      options.onOutput(string, child);

      result.output.push(string);
    });

    child.stderr.on('data', function (data) {
      var string = data.toString();

      options.onError(string, child);

      result.errors.push(string);
    });

    child.on('close', function (code, signal) {
      result.code = code;
      result.signal = signal;

      if (code === 0) {
        resolve(result);
      } else {
        reject(result);
      }
    });
  });
};
