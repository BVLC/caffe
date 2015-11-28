/*jshint node:true*/

var stringUtil = require('ember-cli-string-utils');

module.exports = {
  description: 'The default blueprint for ember-cli projects.',

  locals: function(options) {
    var entity    = options.entity;
    var rawName   = entity.name;
    var name      = stringUtil.dasherize(rawName);
    var namespace = stringUtil.classify(rawName);

    return {
      name: name,
      modulePrefix: name,
      namespace: namespace,
      emberCLIVersion: require('../../package').version
    };
  }
};
