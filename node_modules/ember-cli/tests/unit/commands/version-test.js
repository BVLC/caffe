'use strict';

var expect         = require('chai').expect;
var EOL            = require('os').EOL;
var commandOptions = require('../../factories/command-options');
var VersionCommand = require('../../../lib/commands/version');

describe('version command', function() {
  var options, command;

  beforeEach(function() {
    options = commandOptions({
      project: {
        isEmberCLIProject: function() {
          return false;
        }
      }
    });

    command = new VersionCommand(options);
  });

  it('reports node, npm, and os versions', function() {
    return command.validateAndRun().then(function() {
      var lines = options.ui.output.split(EOL);
      expect(someLineStartsWith(lines, 'node:'), 'contains the version of node');
      expect(someLineStartsWith(lines, 'npm:'), 'contains the version of npm');
      expect(someLineStartsWith(lines, 'os:'), 'contains the version of os');
    });
  });

  it('supports a --verbose flag', function() {
    return command.validateAndRun(['--verbose']).then(function() {
      var lines = options.ui.output.split(EOL);
      expect(someLineStartsWith(lines, 'node:'), 'contains the version of node');
      expect(someLineStartsWith(lines, 'npm:'), 'contains the version of npm');
      expect(someLineStartsWith(lines, 'os:'), 'contains the version of os');
      expect(someLineStartsWith(lines, 'v8:'), 'contains the version of v8');
    });
  });
});

function someLineStartsWith(lines, search) {
  return lines.some(function(line) {
    return line.indexOf(search) === 0;
  });
}
