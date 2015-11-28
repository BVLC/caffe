/*jshint node:true*/

var ModelBlueprint = require('../model');
var testInfo = require('ember-cli-test-info');

module.exports = {
  description: 'Generates a model unit test.',

  locals: function(options) {
    var result = ModelBlueprint.locals.apply(this, arguments);

    result.friendlyDescription = testInfo.description(options.entity.name, "Unit", "Model");

    return result;
  }
};
