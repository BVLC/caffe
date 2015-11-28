'use strict';

var chalk   = require('chalk');
var Task    = require('./task');
var debug   = require('debug')('ember-cli:watcher');
var Promise = require('../ext/promise');
var exec    = Promise.denodeify(require('child_process').exec);
var isWin = /^win/.test(process.platform);

var Watcher = Task.extend({
  verbose: true,

  init: function() {
    var options = this.buildOptions();

    debug('initialize %o', options);

    this.watcher = this.watcher || new (require('broccoli-sane-watcher'))(this.builder, options);

    this.watcher.on('error', this.didError.bind(this));
    this.watcher.on('change', this.didChange.bind(this));
  },

  didError: function(error) {
    debug('didError %o', error);
    this.ui.writeError(error);
    this.analytics.trackError({
      description: error && error.message
    });
  },

  then: function() {
    return this.watcher.then.apply(this.watcher, arguments);
  },

  didChange: function(results) {
    debug('didChange %o', results);
    var totalTime = results.totalTime / 1e6;

    this.ui.writeLine('');
    this.ui.writeLine(chalk.green('Build successful - ' + Math.round(totalTime) + 'ms.'));

    this.analytics.track({
      name:    'ember rebuild',
      message: 'broccoli rebuild time: ' + totalTime + 'ms'
    });

    this.analytics.trackTiming({
      category: 'rebuild',
      variable: 'rebuild time',
      label:    'broccoli rebuild time',
      value:    parseInt(totalTime, 10)
    });
  },

  on: function() {
    this.watcher.on.apply(this.watcher, arguments);
  },

  off: function() {
    this.watcher.off.apply(this.watcher, arguments);
  },
  buildOptions: function() {
    var watcher = this.options && this.options.watcher;

    if (watcher && ['polling', 'watchman', 'node', 'events'].indexOf(watcher) === -1) {
      throw new Error('Unknown watcher type --watcher=[polling|watchman|node] but was: ' + watcher);
    }

    return {
      verbose:  this.verbose,
      poll:     watcher === 'polling',
      watchman: watcher === 'watchman' || watcher === 'events',
      node:     watcher === 'node'
    };
  }
});

Watcher.detectWatcher = function(ui, _options) {
  var options = _options || {};
  var watchmanInfo = 'Visit http://www.ember-cli.com/user-guide/#watchman for more info.';

  if (options.watcher === 'polling') {
    debug('skip detecting watchman, poll instead');
    return Promise.resolve(options);
  } else if (options.watcher === 'node') {
    debug('skip detecting watchman, node instead');
    return Promise.resolve(options);
  } else if (isWin) {
    debug('watchman isn\'t supported on windows, node instead');
    options.watcher = 'node';
    return Promise.resolve(options);
  } else {
    debug('detecting watchman');
    return exec('watchman version').then(function(output) {
      var version;
      try {
        version = JSON.parse(output).version;
      } catch(e) {
        options.watcher = 'node';
        ui.writeLine('Looks like you have a different program called watchman, falling back to NodeWatcher.');
        ui.writeLine(watchmanInfo);
        return options;
      }
      debug('detected watchman: %s', version);

      var semver = require('semver');
      if (semver.satisfies(version, '>= 3.0.0')) {
        debug('watchman %s does satisfy: %s', version, '>= 3.0.0');
        options.watcher = 'watchman';
        options._watchmanInfo = {
          enabled: true,
          version: version,
          canNestRoots: semver.satisfies(version, '>= 3.7.0')
        };
      } else {
        debug('watchman %s does NOT satisfy: %s', version, '>= 3.0.0');
        ui.writeLine('Invalid watchman found, version: [' + version + '] did not satisfy [>= 3.0.0], falling back to NodeWatcher.');
        ui.writeLine(watchmanInfo);
        options.watcher = 'node';
      }

      return options;
    }, function(reason) {
      debug('detecting watchman failed %o', reason);
      ui.writeLine('Could not find watchman, falling back to NodeWatcher for file system events.');
      ui.writeLine(watchmanInfo);
      options.watcher = 'node';
      return options;
    });
  }
};

module.exports = Watcher;
