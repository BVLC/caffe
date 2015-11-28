'use strict';

// Runs `npm install` in cwd

var chalk = require('chalk');
var Task  = require('../models/task');
var npm   = require('../utilities/npm');

module.exports = Task.extend({
  // The command to run: can be 'install' or 'uninstall'
  command: '',
  // Message to send to ui.startProgress
  startProgressMessage: '',
  // Message to send to ui.writeLine on completion
  completionMessage: '',

  init: function() {
    this.npm = this.npm || require('npm');
  },
  // Options: Boolean verbose
  run: function(options) {
    this.ui.startProgress(chalk.green(this.startProgressMessage), chalk.green('.'));

    var npmOptions = {
      loglevel: options.verbose ? 'verbose' : 'error',
      logstream: this.ui.outputStream,
      color: 'always',
      // by default, do install peoples optional deps
      'optional': 'optional' in options ? options.optional : true,
      'save-dev': !!options['save-dev'],
      'save-exact': !!options['save-exact']
    };

    var packages = options.packages || [];

    // npm otherwise is otherwise noisy, already submitted PR for npm to fix
    // misplaced console.log
    this.disableLogger();

    return npm(this.command, packages, npmOptions, this.npm).
      finally(this.finally.bind(this)).
      then(this.announceCompletion.bind(this));
  },

  announceCompletion: function() {
    this.ui.writeLine(chalk.green(this.completionMessage));
  },

  finally: function() {
    this.ui.stopProgress();
    this.restoreLogger();
  },

  disableLogger: function() {
    this.oldLog = console.log;
    console.log = function() {};
  },

  restoreLogger: function() {
    console.log = this.oldLog; // Hack, see above
  }
});
