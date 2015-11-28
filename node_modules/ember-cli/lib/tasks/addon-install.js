'use strict';

var Task                = require('../models/task');
var SilentError         = require('silent-error');
var merge               = require('lodash/object/merge');
var getPackageBaseName  = require('../utilities/get-package-base-name');
var Promise             = require('../ext/promise');

module.exports = Task.extend({
  init: function() {
    this.NpmInstallTask = this.NpmInstallTask || require('./npm-install');
    this.BlueprintTask = this.BlueprintTask || require('./generate-from-blueprint');
  },

  run: function(options) {
    var chalk            = require('chalk');
    var ui               = this.ui;
    var packageNames     = options['packages'];
    var blueprintOptions = options.blueprintOptions || {};

    var npmInstall = new this.NpmInstallTask({
      ui:         this.ui,
      analytics:  this.analytics,
      project:    this.project
    });

    var blueprintInstall = new this.BlueprintTask({
      ui:         this.ui,
      analytics:  this.analytics,
      project:    this.project,
      testing:    this.testing
    });

    ui.startProgress(chalk.green('Installing addon package'), chalk.green('.'));

    return npmInstall.run({
      packages: packageNames,
      'save-dev': true,
      'save-exact': true
    }).then(function() {
      return this.project.reloadAddons();
    }.bind(this)).then(function() {
      return this.installBlueprint(blueprintInstall, packageNames, blueprintOptions);
    }.bind(this))
    .finally(function() { ui.stopProgress(); })
    .then(function() {
      ui.writeLine(chalk.green('Installed addon package.'));
    });
  },

  installBlueprint: function(install, packageNames, blueprintOptions) {
    var blueprintName, taskOptions, addonInstall = this;

    return packageNames.reduce(function(promise, packageName) {
      return promise.then(function() {
        blueprintName = addonInstall.findDefaultBlueprintName(packageName);
        taskOptions = merge({
          args: [blueprintName],
          ignoreMissingMain: true
        }, blueprintOptions || {});
        return install.run(taskOptions);
      });
    }, Promise.resolve());
  },

  findDefaultBlueprintName: function(givenName) {
    var addon = this.project.findAddonByName(givenName);

    if (!addon) {
      throw new SilentError('Install failed. Could not find addon with name: ' + givenName);
    }

    var emberAddon = addon.pkg['ember-addon'];

    if (emberAddon && emberAddon.defaultBlueprint) {
      return emberAddon.defaultBlueprint;
    }

    return getPackageBaseName(addon.pkg.name);
  }
});
