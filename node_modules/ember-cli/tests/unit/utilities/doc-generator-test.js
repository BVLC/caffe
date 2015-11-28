'use strict';

var DocGenerator = require('../../../lib/utilities/doc-generator.js');
var versionUtils = require('../../../lib/utilities/version-utils');
var calculateVersion = versionUtils.emberCLIVersion;
var expect = require('chai').expect;
var path = require('path');
var escapeRegExp = require('escape-string-regexp');

describe('generateDocs', function(){
  it('calls the the appropriate command', function(){
    function execFunc() {
      var commandPath;

      if (process.platform === 'win32') {
        commandPath = escapeRegExp(path.normalize('/node_modules/.bin/yuidoc'));
      } else {
        commandPath = '/node_modules/yuidocjs/lib/cli.js';
      }

      var version = escapeRegExp(calculateVersion());
      var pattern = 'cd docs && .+' + commandPath + ' -q --project-version ' + version;

      expect(arguments[0], 'yudoc command').to.match(new RegExp(pattern));
    }

    new DocGenerator({
      exec: execFunc
    }).generate();
  });
});
