'use strict';

var Promise      = require('../ext/promise');
var versionUtils = require('../utilities/version-utils');
var chalk        = require('chalk');
var debug        = require('debug')('ember-cli:update-checker');

module.exports = UpdateChecker;

function UpdateChecker(ui, settings, localVersion) {
  this.ui = ui;
  this.settings = settings;
  this.localVersion = localVersion || versionUtils.emberCLIVersion();
  this.versionConfig = null;

  debug('version: %s', this.localVersion);
  debug('version: %o', this.settings);
}

/**
* Checks local config or npm for most recent version of ember-cli
*/
UpdateChecker.prototype.checkForUpdates = function() {
  // if 'checkForUpdates' is true, check for an updated ember-cli version
  debug('checkingcheckForUpdates: %o', this.settings.checkingcheckForUpdates);
  if (this.settings.checkForUpdates) {
    return this.doCheck().then(function(updateInfo) {
      debug('updatedNeeded %o', updateInfo.updateNeeded);
      if (updateInfo.updateNeeded) {
        this.ui.writeLine('');
        this.ui.writeLine('A new version of ember-cli is available (' +
                          updateInfo.newestVersion + ').');
      }
      return updateInfo;
    }.bind(this));
  } else {
    return Promise.resolve({
      updateNeeded: false
    });
  }
};

UpdateChecker.prototype.doCheck = function() {
  this.versionConfig = this.versionConfig || new (require('configstore'))('ember-cli-version');
  var lastVersionCheckAt = this.versionConfig.get('lastVersionCheckAt');
  var now = new Date().getTime();

  return new Promise(function(resolve, reject) {
    // if the last check was less than a day ago, don't remotely check version
    if (lastVersionCheckAt && lastVersionCheckAt > (now - 86400000)) {
      resolve(this.versionConfig.get('newestVersion'));
    }

    reject();
  }.bind(this)).catch(function() {
    return this.checkNPM();
  }.bind(this)).then(function(version) {
    var isDevBuild   = versionUtils.isDevelopment(this.localVersion);
    var updateNeeded = false;

    if (!isDevBuild) {
      updateNeeded = version && require('semver').lt(this.localVersion, version);
    }

    return {
      updateNeeded: updateNeeded,
      newestVersion: version
    };
  }.bind(this));
};

UpdateChecker.prototype.checkNPM = function() {
  // use npm to determine the currently availabe ember-cli version
  var loadNPM = Promise.denodeify(require('npm').load);
  var fetchEmberCliVersion = function(npm){
    return Promise.denodeify(npm.commands.view)(['ember-cli', 'version']);
  };
  var extractVersion = function(data) { return Object.keys(data)[0]; };

  return loadNPM({
      'loglevel': 'silent',
      'fetch-retries': 1,
      'cache-min': 1,
      'cache-max': 0
    })
    .then(fetchEmberCliVersion)
    .then(extractVersion)
    .then(this.saveVersionInformation.bind(this))
    .catch(function(){
      this.ui.writeLine(chalk.red('An error occurred while checking for Ember CLI updates. ' +
        'Please verify your internet connectivity and npm configurations.'));
      return false;
    }.bind(this));
};

UpdateChecker.prototype.saveVersionInformation = function(version) {
  var versionConfig = this.versionConfig;
  var now = new Date().getTime();

  // save version so we don't have to check again for another day
  versionConfig.set('newestVersion', version);
  versionConfig.set('lastVersionCheckAt', now);
};
