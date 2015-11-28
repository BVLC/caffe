'use strict';

var lookupCommand  = require('./lookup-command');
var Promise        = require('../ext/promise');
var versionUtils   = require('../utilities/version-utils');
var UpdateChecker  = require('../models/update-checker');
var getOptionArgs  = require('../utilities/get-option-args');
var debug          = require('debug')('ember-cli:cli');
var deprecate      = require('../utilities/deprecate');

var PlatformChecker = require('../utilities/platform-checker');
var emberCLIVersion      = versionUtils.emberCLIVersion;
var InstallationChecker  = require('../models/installation-checker');

function CLI(options) {
  this.name = options.name;
  this.ui = options.ui;
  this.analytics = options.analytics;
  this.testing = options.testing;
  this.root = options.root;
  this.npmPackage = options.npmPackage;

  debug('testing %o', !!this.testing);
}

module.exports = CLI;

CLI.prototype.run = function(environment) {
  return Promise.hash(environment).then(function(environment) {
    var args = environment.cliArgs.slice();
    var commandName = args.shift();
    var commandArgs = args;
    var helpOptions;
    var update;

    var CurrentCommand = lookupCommand(environment.commands, commandName, commandArgs, {
      project: environment.project,
      ui: this.ui
    });

    var command = new CurrentCommand({
      ui:        this.ui,
      analytics: this.analytics,
      commands:  environment.commands,
      tasks:     environment.tasks,
      project:   environment.project,
      settings:  environment.settings,
      testing:   this.testing,
      cli: this
    });

    getOptionArgs('--verbose', commandArgs).forEach(function(arg){
      process.env['EMBER_VERBOSE_' + arg.toUpperCase()] = 'true';
    });

    var platform = new PlatformChecker(process.version);
    if(!platform.isValid && !this.testing) {
      if (platform.isDeprecated) {
        this.ui.writeLine(deprecate('Node ' + process.version + ' is no longer supported by Ember CLI. Please update to a more recent version of Node', true));
      }
      if (platform.isUntested) {
        var chalk = require('chalk');
        this.ui.writeLine(chalk.yellow('WARNING: Node ' + process.version + ' has currently not been tested against Ember CLI and may result in unexpected behaviour.'));
      }
    }

    this.ui.writeLine('version: ' + emberCLIVersion());
    debug('command: %s', commandName);

    if (commandName !== 'update' && !this.testing) {
      var a = new UpdateChecker(this.ui, environment.settings);
      update = a.checkForUpdates();
    }

    if(!this.testing) {
      process.chdir(environment.project.root);
      var skipInstallationCheck = commandArgs.indexOf('--skip-installation-check') !== -1;
      if (environment.project.isEmberCLIProject() && !skipInstallationCheck) {
        new InstallationChecker({ project: environment.project }).checkInstallations();
      }
    }

    command.beforeRun(commandArgs);

    return Promise.resolve(update).then(function() {
      return command.validateAndRun(commandArgs);
    }).then(function(result) {
      // if the help option was passed, call the help command
      if (result === 'callHelp') {
        helpOptions = {
          environment: environment,
          commandName: commandName,
          commandArgs: commandArgs
        };

        return this.callHelp(helpOptions);
      }

      return result;
    }.bind(this)).then(function(exitCode) {
      // TODO: fix this
      // Possibly this issue: https://github.com/joyent/node/issues/8329
      // Wait to resolve promise when running on windows.
      // This ensures that stdout is flushed so acceptance tests get full output
      var result = {
        exitCode: exitCode,
        ui: this.ui
      };
      return new Promise(function(resolve) {
        if (process.platform === 'win32') {
          setTimeout(resolve, 250, result);
        } else {
          resolve(result);
        }
      });
    }.bind(this));

  }.bind(this)).catch(this.logError.bind(this));
};

CLI.prototype.callHelp = function(options) {
  var environment = options.environment;
  var commandName = options.commandName;
  var commandArgs = options.commandArgs;
  var helpIndex = commandArgs.indexOf('--help');
  var hIndex = commandArgs.indexOf('-h');

  var HelpCommand = lookupCommand(environment.commands, 'help', commandArgs, {
    project: environment.project,
    ui: this.ui
  });

  var help = new HelpCommand({
    ui:        this.ui,
    analytics: this.analytics,
    commands:  environment.commands,
    tasks:     environment.tasks,
    project:   environment.project,
    settings:  environment.settings,
    testing:   this.testing
  });

  if (helpIndex > -1) {
    commandArgs.splice(helpIndex,1);
  }

  if (hIndex > -1) {
    commandArgs.splice(hIndex,1);
  }

  commandArgs.unshift(commandName);

  return help.validateAndRun(commandArgs);
};

CLI.prototype.logError = function(error) {
  if (this.testing && error){
    console.error(error.message);
    console.error(error.stack);
  }
  this.ui.writeError(error);
  return 1;
};
