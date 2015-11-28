'use strict';

var chalk    = require('chalk');
var Task     = require('../models/task');
var Builder  = require('../models/builder');

module.exports = Task.extend({
  // Options: String outputPath
  run: function(options) {
    var ui        = this.ui;
    var analytics = this.analytics;

    ui.startProgress(chalk.green('Building'), chalk.green('.'));

    var builder = new Builder({
      ui: ui,
      outputPath: options.outputPath,
      environment: options.environment,
      project: this.project
    });

    return builder.build()
      .then(function(results) {
        var totalTime = results.totalTime / 1e6;

        analytics.track({
          name:    'ember build',
          message: totalTime + 'ms'
        });

        analytics.trackTiming({
          category: 'rebuild',
          variable: 'build time',
          label:    'broccoli build time',
          value:    parseInt(totalTime, 10)
        });
      })
      .finally(function() {
        ui.stopProgress();
        return builder.cleanup();
      })
      .then(function() {
        ui.writeLine(chalk.green('Built project successfully. Stored in "' +
          options.outputPath + '".'));
      })
      .catch(function(err) {
        ui.writeLine(chalk.red('Build failed.'));

        throw err;
      });
  }
});
