'use strict';

// Main entry point
var Project       = require('../models/project');
var requireAsHash = require('../utilities/require-as-hash');
var Command       = require('../models/command');
var commands      = requireAsHash('../commands/*.js', Command);
var Task          = require('../models/task');
var tasks         = requireAsHash('../tasks/*.js', Task);
var CLI           = require('./cli');
var packageConfig = require('../../package.json');
var debug         = require('debug')('ember-cli:cli/index');
var merge         = require('lodash/object/merge');
var path          = require('path');

var version      = packageConfig.version;
var name         = packageConfig.name;
var trackingCode = packageConfig.trackingCode;

function clientId() {
  var ConfigStore = require('configstore');
  var configStore = new ConfigStore('ember-cli');
  var id = configStore.get('client-id');

  if (id) {
    return id;
  } else {
    id = require('node-uuid').v4().toString();
    configStore.set('client-id', id);
    return id;
  }
}

// Options: Array cliArgs, Stream inputStream, Stream outputStream
module.exports = function(options) {
  var UI = options.UI || require('../ui');
  var Leek = options.Leek || require('leek');
  var Yam = options.Yam || require('yam');

  // TODO: one UI (lib/models/project.js also has one for now...)
  var ui = new UI({
    inputStream:  options.inputStream,
    outputStream: options.outputStream,
    errorStream:  options.errorStream || process.stderr,
    ci:           process.env.CI || /^(dumb|emacs)$/.test(process.env.TERM),
    writeLevel:   ~process.argv.indexOf('--silent') ? 'ERROR' : undefined
  });

  var config = new Yam('ember-cli', {
    primary: Project.getProjectRoot()
  });

  var leekOptions;

  var disableAnalytics = options.cliArgs &&
    (options.cliArgs.indexOf('--disable-analytics') > -1 ||
    options.cliArgs.indexOf('-v') > -1 ||
    options.cliArgs.indexOf('--version') > -1) ||
    config.get('disableAnalytics');

  var defaultLeekOptions = {
    trackingCode: trackingCode,
    globalName:   name,
    name:         clientId(),
    version:      version,
    silent:       disableAnalytics
  };

  var defaultUpdateCheckerOptions = {
    checkForUpdates: false
  };

  if (config.get('leekOptions')) {
    leekOptions = merge(defaultLeekOptions, config.get('leekOptions'));
  } else {
    leekOptions = defaultLeekOptions;
  }

  debug('leek: %o', leekOptions);

  var leek = new Leek(leekOptions);

  var cli = new CLI({
    ui:        ui,
    analytics: leek,
    testing:   options.testing,
    name: options.cli ? options.cli.name : 'ember',
    root: options.cli ? options.cli.root : path.resolve(__dirname, '..', '..'),
    npmPackage: options.cli ? options.cli.npmPackage : 'ember-cli'
  });

  var project = Project.projectOrnullProject(ui, cli);

  var environment = {
    tasks:    tasks,
    cliArgs:  options.cliArgs,
    commands: commands,
    project:  project,
    settings: merge(defaultUpdateCheckerOptions, config.getAll())
  };

  return cli.run(environment);
};
