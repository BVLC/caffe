/*jshint node:true*/

var testInfo = require('ember-cli-test-info');

module.exports = {
  description: 'Generates a controller unit test.',
  locals: function(options) {
    return {
      friendlyTestDescription: testInfo.description(options.entity.name, "Unit", "Controller")
    };
  }
};
