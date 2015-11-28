/*jshint node:true*/

var stringUtil  = require('ember-cli-string-utils');
var SilentError = require('silent-error');
var pathUtil    = require('ember-cli-path-utils');

module.exports = {
  description: 'Generates an ember-data adapter.',

  availableOptions: [
    { name: 'base-class', type: String }
  ],

  locals: function(options) {
    var adapterName     = options.entity.name;
    var baseClass       = 'DS.RESTAdapter';
    var importStatement = 'import DS from \'ember-data\';';
    var isAddon         = options.inRepoAddon || options.project.isEmberCLIAddon();
    var relativePath    = pathUtil.getRelativePath(options.entity.name);

    if (options.pod && options.podPath) {
        relativePath = pathUtil.getRelativePath(options.podPath + options.entity.name);
    }

    if (!isAddon && !options.baseClass && adapterName !== 'application') {
      options.baseClass = 'application';
    }

    if (options.baseClass === adapterName) {
      throw new SilentError('Adapters cannot extend from themself. To resolve this, remove the `--base-class` option or change to a different base-class.');
    }

    if (options.baseClass) {
      baseClass = stringUtil.classify(options.baseClass.replace('\/', '-'));
      baseClass = baseClass + 'Adapter';
      importStatement = 'import ' + baseClass + ' from \'' + relativePath + options.baseClass + '\';';
    }

    return {
      importStatement: importStatement,
      baseClass: baseClass
    };
  }
};
