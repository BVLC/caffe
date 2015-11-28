'use strict';

var Command     = require('../models/command');
var Watcher     = require('../models/watcher');
var Builder     = require('../models/builder');
var SilentError = require('silent-error');
var path        = require('path');
var win         = require('../utilities/windows-admin');
var existsSync  = require('exists-sync');

var defaultPort = 7357;

module.exports = Command.extend({
  name: 'test',
  description: 'Runs your app\'s test suite.',
  aliases: ['t'],

  availableOptions: [
    { name: 'environment', type: String,  default: 'test',          aliases: ['e'] },
    { name: 'config-file', type: String,  default: './testem.json', aliases: ['c', 'cf'] },
    { name: 'server',      type: Boolean, default: false,           aliases: ['s'] },
    { name: 'host',        type: String,                            aliases: ['H'] },
    { name: 'test-port',   type: Number,  default: defaultPort,     aliases: ['tp'], description: 'The test port to use when running with --server.' },
    { name: 'filter',      type: String,                            aliases: ['f'],  description: 'A string to filter tests to run' },
    { name: 'module',      type: String,                            aliases: ['m'],  description: 'The name of a test module to run' },
    { name: 'watcher',     type: String,  default: 'events',        aliases: ['w'] },
    { name: 'launch',      type: String,  default: false,                            description: 'A comma separated list of browsers to launch for tests.' },
    { name: 'reporter',    type: String,                            aliases: ['r'],  description: 'Test reporter to use [tap|dot|xunit]' },
    { name: 'test-page',   type: String,                                             description: 'Test page to invoke' },
    { name: 'path',        type: String,                                             description: 'Path to a build to run tests against.' },
    { name: 'query',       type: String,                                             description: 'A query string to append to the test page URL.' }
  ],

  init: function() {
    this.assign    = require('lodash/object/assign');
    this.quickTemp = require('quick-temp');

    this.Builder = this.Builder || Builder;
    this.Watcher = this.Watcher || Watcher;

    if (!this.testing) {
      process.env.EMBER_CLI_TEST_COMMAND = true;
    }
  },

  tmp: function() {
    return this.quickTemp.makeOrRemake(this, '-testsDist');
  },

  rmTmp: function() {
    this.quickTemp.remove(this, '-testsDist');
    this.quickTemp.remove(this, '-customConfigFile');
  },

  _generateCustomConfigs: function(options) {
    var config = {};
    if (!options.filter && !options.module && !options.launch && !options.query && !options['test-page']) { return config; }

    var testPage = options['test-page'];
    var queryString = this.buildTestPageQueryString(options);
    if (testPage) {
      var containsQueryString = testPage.indexOf('?') > -1;
      var testPageJoinChar    = containsQueryString ? '&' : '?';
      config.testPage = testPage + testPageJoinChar + queryString;
    }
    if (queryString) {
      config.queryString = queryString;
    }

    if (options.launch) {
      config.launch = options.launch;
    }

    return config;
  },

  _generateTestPortNumber: function(options) {
    if (options.port && options.testPort !== defaultPort || !isNaN(parseInt(options.testPort)) && !options.port) { return options.testPort; }
    if (options.port) { return parseInt(options.port, 10) + 1; }
  },

  buildTestPageQueryString: function(options) {
    var params = [];

    if (options.module) {
      params.push('module=' + options.module);
    }

    if (options.filter) {
      params.push('filter=' + options.filter.toLowerCase());
    }

    if (options.query) {
      params.push(options.query);
    }

    return params.join('&');
  },

  run: function(commandOptions) {
    var hasBuild = !!commandOptions.path;
    var outputPath;

    if (hasBuild) {
      outputPath = path.resolve(commandOptions.path);

      if (!existsSync(outputPath)) {
        throw new SilentError('The path ' + commandOptions.path + ' does not exist. Please specify a valid build directory to test.');
      }
    } else {
      outputPath = this.tmp();
    }

    process.env['EMBER_CLI_TEST_OUTPUT'] = outputPath;
    var testOptions = this.assign({}, commandOptions, {
      outputPath: outputPath,
      project: this.project,
      port: this._generateTestPortNumber(commandOptions)
    }, this._generateCustomConfigs(commandOptions));

    var options = {
      ui: this.ui,
      analytics: this.analytics,
      project: this.project
    };

    if (commandOptions.server) {
      if (hasBuild) {
        throw new SilentError('Specifying a build is not allowed with the `--server` option.');
      }

      options.builder = new this.Builder(testOptions);

      var TestServerTask = this.tasks.TestServer;
      var testServer     = new TestServerTask(options);

      testOptions.watcher = new this.Watcher(this.assign(options, {
        verbose: false,
        options: commandOptions
      }));

      return testServer.run(testOptions)
        .finally(this.rmTmp.bind(this));
    } else {
      var TestTask  = this.tasks.Test;
      var test  = new TestTask(options);

      if (hasBuild) {
        return win.checkWindowsElevation(this.ui).then(function() {
          return test.run(testOptions).finally(this.rmTmp.bind(this));
        }.bind(this));
      }

      var BuildTask = this.tasks.Build;
      var build = new BuildTask(options);

      return win.checkWindowsElevation(this.ui).then(function() {
        return build.run({
          environment: commandOptions.environment,
          outputPath: outputPath
        })
        .then(function() {
          return test.run(testOptions);
        })
        .finally(this.rmTmp.bind(this));
      }.bind(this));
    }
  }
});
