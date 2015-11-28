/*jshint node:true*/

var getDependencyDepth = require('ember-cli-get-dependency-depth');
var testInfo = require('ember-cli-test-info');

module.exports = {
  description: 'Generates an initializer unit test.',
  locals: function(options) {
    return {
      friendlyTestName: testInfo.name(options.entity.name, "Unit", "Initializer"),
      dependencyDepth: getDependencyDepth(options.entity.name)
    };
  }
};
