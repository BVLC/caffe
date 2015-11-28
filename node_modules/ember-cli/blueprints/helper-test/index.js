/*jshint node:true*/

var getDependencyDepth = require('ember-cli-get-dependency-depth');
var testInfo = require('ember-cli-test-info');

module.exports = {
  description: 'Generates a helper unit test.',
  locals: function(options) {
    return {
      friendlyTestName: testInfo.name(options.entity.name, "Unit", "Helper"),
      dependencyDepth: getDependencyDepth(options.entity.name)
    };
  }
};
