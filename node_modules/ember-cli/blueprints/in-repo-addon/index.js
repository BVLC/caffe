/*jshint node:true*/

var fs   = require('fs-extra');
var path = require('path');
var stringUtil = require('ember-cli-string-utils');
var Blueprint = require('../../lib/models/blueprint');

module.exports = {
  description: 'The blueprint for addon in repo ember-cli addons.',

  beforeInstall: function(options) {
    var libBlueprint = Blueprint.lookup('lib', {
      ui: this.ui,
      analytics: this.analytics,
      project: this.project
    });

    return libBlueprint.install(options);
  },

  afterInstall: function(options) {
    var packagePath = path.join(this.project.root, 'package.json');
    var contents    = JSON.parse(fs.readFileSync(packagePath, { encoding: 'utf8' }));
    var name        = stringUtil.dasherize(options.entity.name);
    var newPath     = ['lib', name].join('/');
    var paths;

    contents['ember-addon'] = contents['ember-addon'] || {};
    paths = contents['ember-addon']['paths'] = contents['ember-addon']['paths'] || [];

    if (paths.indexOf(newPath) === -1) {
      paths.push(newPath);
    }

    fs.writeFileSync(packagePath, JSON.stringify(contents, null, 2));
  }
};
