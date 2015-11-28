'use strict';

var debug       = require('debug')('ember-cli:installation-checker');
var fs          = require('fs');
var existsSync  = require('exists-sync');
var path        = require('path');
var SilentError = require('silent-error');

module.exports = InstallationChecker;

function InstallationChecker(options) {
  this.project = options.project;
}

/**
* Check if npm and bower installation directories are present,
* and raise an error message with instructions on how to proceed.
*
* If some of these package managers aren't being used in the project
* we just ignore them. Their usage is considered by checking the
* presence of your manifest files: package.json for npm and bower.json for bower.
*/
InstallationChecker.prototype.checkInstallations = function() {
  var commands = [];

  if (this.usingNpm() && this.npmDependenciesNotPresent()) {
    debug('npm dependencies not installed');
    commands.push('`npm install`');
  }
  if (this.usingBower() && this.bowerDependenciesNotPresent()) {
    debug('bower dependencies not installed');
    commands.push('`bower install`');
  }
  if (commands.length) {
    var commandText = commands.join(' and ');
    throw new SilentError('No dependencies installed. Run ' + commandText + ' to install missing dependencies.');
  }
};

function hasDependencies(pkg) {
  return (pkg.dependencies && pkg.dependencies.length) ||
         (pkg.devDependencies && pkg.devDependencies.length);
}

function readJSON(path) {
  try {
    return JSON.parse(fs.readFileSync(path).toString());
  } catch(e) {
    throw new SilentError('InstallationChecker: Unable to parse: ' + path);
  }
}

InstallationChecker.prototype.hasBowerDeps = function() {
  return hasDependencies(readJSON(path.join(this.project.root, 'bower.json')));
};

InstallationChecker.prototype.usingBower = function() {
  return existsSync(path.join(this.project.root, 'bower.json')) && this.hasBowerDeps();
};

InstallationChecker.prototype.bowerDependenciesNotPresent = function() {
  return !existsSync(this.project.bowerDirectory);
};

InstallationChecker.prototype.hasNpmDeps = function() {
  return hasDependencies(readJSON(path.join(this.project.root, 'package.json')));
};

InstallationChecker.prototype.usingNpm = function() {
  return existsSync(path.join(this.project.root, 'package.json')) && this.hasNpmDeps();
};

InstallationChecker.prototype.npmDependenciesNotPresent = function() {
  return !existsSync(this.project.nodeModulesPath);
};
