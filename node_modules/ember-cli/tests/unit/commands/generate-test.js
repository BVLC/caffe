/*jshint multistr: true */

'use strict';

var expect            = require('chai').expect;
var EOL               = require('os').EOL;
var SilentError       = require('silent-error');
var commandOptions    = require('../../factories/command-options');
var stub              = require('../../helpers/stub').stub;
var processHelpString = require('../../helpers/process-help-string');
var MockProject       = require('../../helpers/mock-project');
var Promise           = require('../../../lib/ext/promise');
var Task              = require('../../../lib/models/task');
var Blueprint         = require('../../../lib/models/blueprint');
var GenerateCommand   = require('../../../lib/commands/generate');

describe('generate command', function() {
  var options, command;

  beforeEach(function() {
    var project = new MockProject();

    project.isEmberCLIProject = function() {
      return true;
    };

    project.blueprintLookupPaths = function() {
      return [];
    };

    options = commandOptions({
      project: project,

      tasks: {
        GenerateFromBlueprint: Task.extend({
          project: project,
          run: function(options) {
            return Promise.resolve(options);
          }
        })
      }
    });

    //nodeModulesPath: 'somewhere/over/the/rainbow'
    command = new GenerateCommand(options);
  });

  afterEach(function() {
    if (Blueprint.list.restore) {
      Blueprint.list.restore();
    }
  });

  it('runs GenerateFromBlueprint but with null nodeModulesPath', function() {
    command.project.hasDependencies = function() { return false; };

    expect(function() {
      command.validateAndRun(['controller', 'foo']);
    }).to.throw(SilentError, 'node_modules appears empty, you may need to run `npm install`');
  });

  it('runs GenerateFromBlueprint with expected options', function() {
    return command.validateAndRun(['controller', 'foo'])
      .then(function(options) {
        expect(options.pod, false);
        expect(options.dryRun, false);
        expect(options.verbose, false);
        expect(options.args).to.deep.equal(['controller', 'foo']);
      });
  });

  it('does not throws errors when beforeRun is invoked without the blueprint name', function() {
    expect(function() {
      command.beforeRun([]);
    }).to.not.throw();
  });

  it('complains if no blueprint name is given', function() {
    return command.validateAndRun([])
      .then(function() {
        expect(false, 'should not have called run');
      })
      .catch(function(error) {
        expect(error.message).to.equal(
            'The `ember generate` command requires a ' +
            'blueprint name to be specified. ' +
            'For more details, use `ember help`.');
      });
  });

  describe('help', function() {
    it('lists available blueprints', function() {
      stub(Blueprint, 'list', [
        {
          source: 'my-app',
          blueprints: [
            {
              name: 'my-blueprint',
              availableOptions: [],
              printBasicHelp: function() {
                return this.name;
              }
            },
            {
              name: 'other-blueprint',
              availableOptions: [],
              printBasicHelp: function() {
                return this.name;
              }
            }
          ]
        }
      ]);

      command.printDetailedHelp({});

      var output = options.ui.output;

      var testString = processHelpString(EOL + '\
  Available blueprints:' + EOL + '\
    my-app:' + EOL + '\
my-blueprint' + EOL + '\
other-blueprint' + EOL + '\
' + EOL);

      expect(output).to.equal(testString);
    });

    it('lists available blueprints json', function() {
      stub(Blueprint, 'list', [
        {
          source: 'my-app',
          blueprints: [
            {
              name: 'my-blueprint',
              availableOptions: [],
              getJson: function() {
                return {
                  name: this.name
                };
              }
            },
            {
              name: 'other-blueprint',
              availableOptions: [],
              getJson: function() {
                return {
                  name: this.name
                };
              }
            }
          ]
        }
      ]);

      var json = {};

      command.addAdditionalJsonForHelp(json, {
        json: true
      });

      expect(json.availableBlueprints).to.deep.equal([
        {
          'my-app': [
            {
              name: 'my-blueprint',
            },
            {
              name: 'other-blueprint',
            }
          ]
        }
      ]);
    });

    it('works with single blueprint', function() {
      stub(Blueprint, 'list', [
        {
          source: 'my-app',
          blueprints: [
            {
              name: 'my-blueprint',
              availableOptions: [],
              printBasicHelp: function() {
                return this.name;
              }
            },
            {
              name: 'skipped-blueprint'
            }
          ]
        }
      ]);

      command.printDetailedHelp({
        rawArgs: ['my-blueprint']
      });

      var output = options.ui.output;

      var testString = processHelpString('\
my-blueprint' + EOL + '\
' + EOL);

      expect(output).to.equal(testString);
    });

    it('works with single blueprint json', function() {
      stub(Blueprint, 'list', [
        {
          source: 'my-app',
          blueprints: [
            {
              name: 'my-blueprint',
              availableOptions: [],
              getJson: function() {
                return {
                  name: this.name
                };
              }
            },
            {
              name: 'skipped-blueprint'
            }
          ]
        }
      ]);

      var json = {};

      command.addAdditionalJsonForHelp(json, {
        rawArgs: ['my-blueprint'],
        json: true
      });

      expect(json.availableBlueprints).to.deep.equal([
        {
          'my-app': [
            {
              name: 'my-blueprint',
            }
          ]
        }
      ]);
    });

    it('handles missing blueprint', function() {
      stub(Blueprint, 'list', [
        {
          source: 'my-app',
          blueprints: [
            {
              name: 'my-blueprint'
            }
          ]
        }
      ]);

      command.printDetailedHelp({
        rawArgs: ['missing-blueprint']
      });

      var output = options.ui.output;

      var testString = processHelpString('\
\u001b[33mThe \'missing-blueprint\' blueprint does not exist in this project.\u001b[39m' + EOL + '\
' + EOL);

      expect(output).to.equal(testString);
    });

    it('handles missing blueprint json', function() {
      stub(Blueprint, 'list', [
        {
          source: 'my-app',
          blueprints: [
            {
              name: 'my-blueprint'
            }
          ]
        }
      ]);

      var json = {};

      command.addAdditionalJsonForHelp(json, {
        rawArgs: ['missing-blueprint'],
        json: true
      });

      expect(json.availableBlueprints).to.deep.equal([
        {
          'my-app': []
        }
      ]);
    });

    it('ignores overridden blueprints when verbose false', function() {
      stub(Blueprint, 'list', [
        {
          source: 'my-app',
          blueprints: [
            {
              name: 'my-blueprint',
              availableOptions: [],
              printBasicHelp: function() {
                return this.name;
              },
              overridden: true
            }
          ]
        }
      ]);

      command.printDetailedHelp({});

      var output = options.ui.output;

      var testString = processHelpString(EOL + '\
  Available blueprints:' + EOL + '\
' + EOL);

      expect(output).to.equal(testString);
    });

    it('shows overridden blueprints when verbose true', function() {
      stub(Blueprint, 'list', [
        {
          source: 'my-app',
          blueprints: [
            {
              name: 'my-blueprint',
              availableOptions: [],
              printBasicHelp: function() {
                return this.name;
              },
              overridden: true
            }
          ]
        }
      ]);

      command.printDetailedHelp({
        verbose: true
      });

      var output = options.ui.output;

      var testString = processHelpString(EOL + '\
  Available blueprints:' + EOL + '\
    my-app:' + EOL + '\
my-blueprint' + EOL + '\
' + EOL);

      expect(output).to.equal(testString);
    });
  });
});
