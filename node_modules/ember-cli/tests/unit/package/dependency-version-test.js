'use strict';

var expect = require('chai').expect;
var semver = require('semver');

function assertVersionLock(_deps) {
  var deps = _deps || {};

  Object.keys(deps).forEach(function(name) {
    if (name !== 'ember-cli' &&
        semver.valid(deps[name]) &&
        semver.gtr('1.0.0', deps[name])) {
      // only valid if the version is fixed
      expect(semver.valid(deps[name]), '"' + name + '" has a valid version');
    }
  });
}

describe('dependencies', function() {
  var pkg;

  describe('in package.json', function() {
    before(function() {
      pkg = require('../../../package.json');
    });

    it('are locked down for pre-1.0 versions', function() {
      assertVersionLock(pkg.dependencies);
      assertVersionLock(pkg.devDependencies);
    });
  });

  describe('in blueprints/app/files/package.json', function() {
    before(function() {
      pkg = require('../../../blueprints/app/files/package.json');
    });

    it('are locked down for pre-1.0 versions', function() {
      assertVersionLock(pkg.dependencies);
      assertVersionLock(pkg.devDependencies);
    });
  });
});
