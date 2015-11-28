/*jshint node:true*/
var isPackageMissing = require('ember-cli-is-package-missing');

module.exports = {
  description: 'Generates a server directory for mocks and proxies.',

  normalizeEntityName: function() {},

  afterInstall: function(options) {

    var isMorganMissing = isPackageMissing(this, 'morgan');
    var isGlobMissing = isPackageMissing(this, 'glob');

    var areDependenciesMissing = isMorganMissing || isGlobMissing;
    var libsToInstall = [];

    if (isMorganMissing) {
      libsToInstall.push({ name: 'morgan', target: '^1.3.2' });
    }

    if (isGlobMissing) {
      libsToInstall.push({ name: 'glob', target: '^4.0.5' });
    }

    if (!options.dryRun && areDependenciesMissing) {
      return this.addPackagesToProject(libsToInstall);
    }
  }
};
