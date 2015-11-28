'use strict';

var path    = require('path');
var Command = require('../models/command');
var win     = require('../utilities/windows-admin');

module.exports = Command.extend({
  name: 'build',
  description: 'Builds your app and places it into the output path (dist/ by default).',
  aliases: ['b'],

  availableOptions: [
    { name: 'environment', type: String,  default: 'development', aliases: ['e', { 'dev': 'development' }, { 'prod': 'production' }] },
    { name: 'output-path', type: path,    default: 'dist/',       aliases: ['o'] },
    { name: 'watch',       type: Boolean, default: false,         aliases: ['w'] },
    { name: 'watcher',     type: String }
  ],

  run: function(commandOptions) {
    var BuildTask = this.taskFor(commandOptions);
    var buildTask = new BuildTask({
      ui: this.ui,
      analytics: this.analytics,
      project: this.project
    });

    return win.checkWindowsElevation(this.ui).then(function() {
      return buildTask.run(commandOptions);
    });
  },

  taskFor: function(options) {
    if (options.watch) {
      return this.tasks.BuildWatch;
    } else {
      return this.tasks.Build;
    }
  }
});
