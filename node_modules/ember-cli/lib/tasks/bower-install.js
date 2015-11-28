'use strict';

// Runs `bower install` in cwd

var Promise = require('../ext/promise');
var Task    = require('../models/task');

module.exports = Task.extend({
  init: function() {
    this.bower = this.bower || require('bower');
    this.bowerConfig = this.bowerConfig || require('bower-config');
  },
  // Options: Boolean verbose
  run: function(options) {
    var chalk          = require('chalk');
    var bower          = this.bower;
    var bowerConfig    = this.bowerConfig;
    var ui             = this.ui;
    var packages       = options.packages || [];
    var installOptions = options.installOptions || { save: true };

    ui.startProgress(chalk.green('Installing browser packages via Bower'), chalk.green('.'));

    var config = bowerConfig.read();
    config.interactive = true;

    return new Promise(function(resolve, reject) {
        bower.commands.install(packages, installOptions, config) // Packages, options, config
          .on('log', logBowerMessage)
          .on('prompt', ui.prompt.bind(ui))
          .on('error', reject)
          .on('end', resolve);
      })
      .finally(function() { ui.stopProgress(); })
      .then(function() {
        ui.writeLine(chalk.green('Installed browser packages via Bower.'));
      });

    function logBowerMessage(message) {
      if (message.level === 'conflict') {
        // e.g.
        //   conflict Unable to find suitable version for ember-data
        //     1) ember-data 1.0.0-beta.6
        //     2) ember-data ~1.0.0-beta.7
        ui.writeLine('  ' + chalk.red('conflict') + ' ' + message.message);
        message.data.picks.forEach(function(pick, index) {
          ui.writeLine('    ' + chalk.green((index + 1) + ')') + ' ' +
                       message.data.name + ' ' + pick.endpoint.target);
        });
      } else if (message.level === 'info' && options.verbose) {
        // e.g.
        //   cached git://example.com/some-package.git#1.0.0
        ui.writeLine('  ' + chalk.green(message.id) + ' ' + message.message);
      }
    }
  }
});
