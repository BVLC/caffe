/*jshint node:true*/

var testInfo    = require('ember-cli-test-info');
var pathUtil    = require('ember-cli-path-utils');
var stringUtils = require('ember-cli-string-utils');

module.exports = {
  description: 'Generates an acceptance test for a feature.',
  locals: function(options) {
    var testFolderRoot = stringUtils.dasherize(options.project.name());

    if (options.project.isEmberCLIAddon()) {
      testFolderRoot = pathUtil.getRelativeParentPath(options.entity.name, -1, false);
    }
    return {
      testFolderRoot: testFolderRoot,
      friendlyTestName: testInfo.name(options.entity.name, "Acceptance", null)
    };
  }
};
