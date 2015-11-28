/*jshint node:true*/

var testInfo = require('ember-cli-test-info');

module.exports = {
  description: 'Generates a route unit test.',
  locals: function(options) {
    return {
      friendlyTestDescription: testInfo.description(options.entity.name, "Unit", "Route")
    };
  },
};
