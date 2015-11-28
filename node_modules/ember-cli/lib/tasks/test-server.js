'use strict';

var TestTask = require('./test');
var Promise  = require('../ext/promise');
var remove   = Promise.denodeify(require('fs-extra').remove);
var chalk    = require('chalk');

module.exports = TestTask.extend({
  init: function() {
    this.testem = this.testem || new (require('testem'))();
  },

  invokeTestem: function(options) {
    var task = this;

    return new Promise(function(resolve) {
      task.testem.startDev(task.testemOptions(options), function(code) {
        remove(options.outputPath)
        .finally(function() {
          resolve(code);
        });
      });
    });
  },

  run: function(options) {
    var ui = this.ui;
    var testem = this.testem;
    var task = this;

    // The building has actually started already, but we want some output while we wait for the server
    ui.startProgress(chalk.green('Building'), chalk.green('.'));

    return new Promise(function(resolve) {
      var watcher = options.watcher;
      var started = false;

      // Wait for a build and then either start or restart testem
      watcher.on('change', function() {
        if (started) {
          testem.restart();
        } else {
          started = true;

          ui.stopProgress();
          resolve(task.invokeTestem(options));
        }
      });
    });
  }
});
