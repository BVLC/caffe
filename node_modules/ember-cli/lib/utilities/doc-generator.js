'use strict';

var versionUtils     = require('./version-utils');
var calculateVersion = versionUtils.emberCLIVersion;
var fs               = require('fs');

function DocGenerator(options) {
  options = options || {};
  this.exec = options.exec || require('child_process').exec;
}

DocGenerator.prototype.generate = function() {
  var command = 'cd docs && ' + fs.realpathSync('./node_modules/.bin/yuidoc') +
                ' -q --project-version ' + calculateVersion(); // add '-p' flag to produce only JSON and not HTML

  console.log('Executing command: ' + command);
  this.exec(command, function(error){ // stdout, stderr
    if (error !== null) {
      console.log('Error: ' + error);
    }
  });
};

module.exports = DocGenerator;
