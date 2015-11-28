/*jshint multistr: true */

'use strict';

var expect            = require('chai').expect;
var EOL               = require('os').EOL;
var proxyquire        = require('proxyquire');
var path              = require('path');
var stub              = require('../../helpers/stub').stub;
var processHelpString = require('../../helpers/process-help-string');
var convertToJson     = require('../../helpers/convert-help-output-to-json');
var commandOptions    = require('../../factories/command-options');

var lookupCommandStub;
var HelpCommand = proxyquire('../../../lib/commands/help', {
  '../cli/lookup-command': function() {
    return lookupCommandStub.apply(this, arguments);
  }
});

describe('help command', function() {
  var options;

  beforeEach(function() {
    options = commandOptions();

    lookupCommandStub = require('../../../lib/cli/lookup-command');
  });

  describe('common to both', function() {
    it('finds command on disk', function() {
      var Command1 = function() {};
      stub(Command1.prototype, 'printBasicHelp');
      stub(Command1.prototype, 'printDetailedHelp');

      options.commands = {
        Command1: Command1
      };

      var wasCalled;
      lookupCommandStub = function() {
        expect(arguments[0]).to.equal(options.commands);
        expect(arguments[1]).to.equal('command-2');
        wasCalled = true;
        return Command1;
      };

      var command = new HelpCommand(options);

      command.run(options, ['command-2']);

      expect(Command1.prototype.printBasicHelp.called).to.equal(1);
      expect(wasCalled).to.be.true;
    });

    it('looks up multiple commands', function() {
      var Command1 = function() {};
      var Command2 = function() {};
      var Command3 = function() {};
      stub(Command1.prototype, 'printBasicHelp');
      stub(Command2.prototype, 'printBasicHelp');
      stub(Command3.prototype, 'printBasicHelp');
      stub(Command1.prototype, 'printDetailedHelp');
      stub(Command2.prototype, 'printDetailedHelp');
      stub(Command3.prototype, 'printDetailedHelp');

      options.commands = {
        Command1: Command1,
        Command2: Command2,
        Command3: Command3
      };

      var command = new HelpCommand(options);

      command.run(options, ['command-1', 'command-2']);

      expect(Command1.prototype.printBasicHelp.called).to.equal(1);
      expect(Command2.prototype.printBasicHelp.called).to.equal(1);
      expect(Command3.prototype.printBasicHelp.called).to.equal(0);
      expect(Command1.prototype.printDetailedHelp.called).to.equal(1);
      expect(Command2.prototype.printDetailedHelp.called).to.equal(1);
      expect(Command3.prototype.printDetailedHelp.called).to.equal(0);
    });
  });

  describe('unique to text printing', function() {
    it('lists commands', function() {
      var Command1 = function() {};
      var Command2 = function() {};
      stub(Command1.prototype, 'printBasicHelp');
      stub(Command2.prototype, 'printBasicHelp');
      stub(Command1.prototype, 'printDetailedHelp');
      stub(Command2.prototype, 'printDetailedHelp');

      options.commands = {
        Command1: Command1,
        Command2: Command2
      };

      var command = new HelpCommand(options);

      command.run(options, []);

      expect(Command1.prototype.printBasicHelp.called).to.equal(1);
      expect(Command2.prototype.printBasicHelp.called).to.equal(1);
      expect(Command1.prototype.printDetailedHelp.called).to.equal(0);
      expect(Command2.prototype.printDetailedHelp.called).to.equal(0);
    });

    it('works with single command', function() {
      var Command1 = function() {};
      var Command2 = function() {};
      stub(Command1.prototype, 'printBasicHelp');
      stub(Command2.prototype, 'printBasicHelp');
      stub(Command1.prototype, 'printDetailedHelp');
      stub(Command2.prototype, 'printDetailedHelp');

      options.commands = {
        Command1: Command1,
        Command2: Command2
      };

      var command = new HelpCommand(options);

      command.run(options, ['command-1']);

      expect(Command1.prototype.printBasicHelp.called).to.equal(1);
      expect(Command2.prototype.printBasicHelp.called).to.equal(0);
      expect(Command1.prototype.printDetailedHelp.called).to.equal(1);
      expect(Command2.prototype.printDetailedHelp.called).to.equal(0);
    });

    it('works with single command alias', function() {
      var Command1 = function() {};
      Command1.prototype.aliases = ['my-alias'];
      stub(Command1.prototype, 'printBasicHelp');
      stub(Command1.prototype, 'printDetailedHelp');

      options.commands = {
        Command1: Command1
      };

      var command = new HelpCommand(options);

      command.run(options, ['my-alias']);

      expect(Command1.prototype.printBasicHelp.called).to.equal(1);
    });

    it('passes extra commands to `generate`', function() {
      var Generate = function() {};
      stub(Generate.prototype, 'printBasicHelp');
      stub(Generate.prototype, 'printDetailedHelp');

      options.commands = {
        Generate: Generate
      };

      var command = new HelpCommand(options);

      command.run(options, ['generate', 'something', 'else']);

      expect(Generate.prototype.printBasicHelp.calledWith[0][0].rawArgs).to.deep.equal(['something', 'else']);
      expect(Generate.prototype.printDetailedHelp.calledWith[0][0].rawArgs).to.deep.equal(['something', 'else']);
    });

    it('handles no extra commands to `generate`', function() {
      var Generate = function() {};
      stub(Generate.prototype, 'printBasicHelp');
      stub(Generate.prototype, 'printDetailedHelp');

      options.commands = {
        Generate: Generate
      };

      var command = new HelpCommand(options);

      command.run(options, ['generate']);

      expect(Generate.prototype.printBasicHelp.calledWith[0][0].rawArgs).to.equal(undefined);
      expect(Generate.prototype.printDetailedHelp.calledWith[0][0].rawArgs).to.equal(undefined);
    });

    it('passes extra commands to `generate` alias', function() {
      var Generate = function() {};
      Generate.prototype.aliases = ['g'];
      stub(Generate.prototype, 'printBasicHelp');
      stub(Generate.prototype, 'printDetailedHelp');

      options.commands = {
        Generate: Generate
      };

      var command = new HelpCommand(options);

      command.run(options, ['g', 'something', 'else']);

      expect(Generate.prototype.printBasicHelp.calledWith[0][0].rawArgs).to.deep.equal(['something', 'else']);
      expect(Generate.prototype.printDetailedHelp.calledWith[0][0].rawArgs).to.deep.equal(['something', 'else']);
    });

    it('handles missing command', function() {
      options.commands = {
        Command1: function() {}
      };

      var command = new HelpCommand(options);

      command.run(options, ['missing-command']);

      var output = options.ui.output;

      var testString = processHelpString('\
Requested ember-cli commands:' + EOL + '\
' + EOL + '\
\u001b[31mNo help entry for \'missing-command\'\u001b[39m' + EOL);

      expect(output).to.include(testString);
    });

    it('respects skipHelp when listing', function() {
      var Command1 = function() { this.skipHelp = true; };
      var Command2 = function() {};
      stub(Command1.prototype, 'printBasicHelp');
      stub(Command2.prototype, 'printBasicHelp');

      options.commands = {
        Command1: Command1,
        Command2: Command2
      };

      var command = new HelpCommand(options);

      command.run(options, []);

      expect(Command1.prototype.printBasicHelp.called).to.equal(0);
      expect(Command2.prototype.printBasicHelp.called).to.equal(1);
    });

    it('ignores skipHelp when single', function() {
      var Command1 = function() { this.skipHelp = true; };
      stub(Command1.prototype, 'printBasicHelp');
      stub(Command1.prototype, 'printDetailedHelp');

      options.commands = {
        Command1: Command1
      };

      var command = new HelpCommand(options);

      command.run(options, ['command-1']);

      expect(Command1.prototype.printBasicHelp.called).to.equal(1);
    });

    it('lists addons', function() {
      var Command1 = function() {};
      var Command2 = function() {};
      stub(Command1.prototype, 'printBasicHelp');
      stub(Command2.prototype, 'printBasicHelp');

      options.project.eachAddonCommand = function(callback) {
        callback('my-addon', {
          Command1: Command1,
          Command2: Command2
        });
      };

      var command = new HelpCommand(options);

      command.run(options, []);

      var output = options.ui.output;

      var testString = processHelpString(EOL + '\
Available commands from my-addon:' + EOL);

      expect(output).to.include(testString);

      expect(Command1.prototype.printBasicHelp.called).to.equal(1);
      expect(Command2.prototype.printBasicHelp.called).to.equal(1);
    });

    it('finds single addon command', function() {
      var Command1 = function() {};
      var Command2 = function() {};
      stub(Command1.prototype, 'printBasicHelp');
      stub(Command1.prototype, 'printDetailedHelp');

      options.project.eachAddonCommand = function(callback) {
        callback('my-addon', {
          Command1: Command1,
          Command2: Command2
        });
      };

      var command = new HelpCommand(options);

      command.run(options, ['command-1']);

      expect(Command1.prototype.printBasicHelp.called).to.equal(1);
    });
  });

  describe('unique to json printing', function() {
    beforeEach(function() {
      options.json = true;
    });

    it('lists commands', function() {
      options.commands = {
        Command1: function() {
          return {
            getJson: function() {
              return {
                test1: 'bar'
              };
            }
          };
        },
        Command2: function() {
          return {
            getJson: function() {
              return {
                test2: 'bar'
              };
            }
          };
        }
      };

      var command = new HelpCommand(options);

      command.run(options, []);

      var json = convertToJson(options.ui.output);

      expect(json.commands).to.deep.equal([
        {
          test1: 'bar'
        },
        {
          test2: 'bar'
        }
      ]);
    });

    it('works with single command alias', function() {
      var Command1 = function() {
        return {
          getJson: function() {
            return {
              test1: 'bar'
            };
          }
        };
      };
      Command1.prototype.aliases = ['my-alias'];

      options.commands = {
        Command1: Command1
      };

      var command = new HelpCommand(options);

      command.run(options, ['my-alias']);

      var json = convertToJson(options.ui.output);

      expect(json.commands).to.deep.equal([
        {
          test1: 'bar'
        }
      ]);
    });

    it('passes extra commands to `generate`', function() {
      options.commands = {
        Generate: function() {
          return {
            getJson: function(options) {
              expect(options.rawArgs).to.deep.equal(['something', 'else']);
              return {
                test1: 'bar'
              };
            }
          };
        }
      };

      var command = new HelpCommand(options);

      command.run(options, ['generate', 'something', 'else']);

      var json = convertToJson(options.ui.output);

      expect(json.commands).to.deep.equal([
        {
          test1: 'bar'
        }
      ]);
    });

    it('handles no extra commands to `generate`', function() {
      options.commands = {
        Generate: function() {
          return {
            getJson: function(options) {
              expect(options.rawArgs).to.equal(undefined);
              return {
                test1: 'bar'
              };
            }
          };
        }
      };

      var command = new HelpCommand(options);

      command.run(options, ['generate']);

      var json = convertToJson(options.ui.output);

      expect(json.commands).to.deep.equal([
        {
          test1: 'bar'
        }
      ]);
    });

    it('passes extra commands to `generate` alias', function() {
      var Generate = function() {
        return {
          getJson: function() {
            return {
              test1: 'bar'
            };
          }
        };
      };
      Generate.prototype.aliases = ['g'];

      options.commands = {
        Generate: Generate
      };

      var command = new HelpCommand(options);

      command.run(options, ['g', 'something', 'else']);

      var json = convertToJson(options.ui.output);

      expect(json.commands).to.deep.equal([
        {
          test1: 'bar'
        }
      ]);
    });

    it('handles special option `path`', function() {
      options.commands = {
        Command1: function() {
          return {
            getJson: function() {
              return {
                test1: path
              };
            }
          };
        }
      };

      var command = new HelpCommand(options);

      command.run(options, ['command-1']);

      var json = convertToJson(options.ui.output);

      expect(json.commands).to.deep.equal([
        {
          test1: 'path'
        }
      ]);
    });

    it('handles missing command', function() {
      options.commands = {
        Command1: function() {}
      };

      var command = new HelpCommand(options);

      command.run(options, ['missing-command']);

      var json = convertToJson(options.ui.output);

      expect(json.commands).to.deep.equal([
        {
          name: 'core-object',
          description: null,
          aliases: [],
          works: 'insideProject',
          availableOptions: [],
          anonymousOptions: []
        }
      ]);
    });

    it('respects skipHelp when listing', function() {
      options.commands = {
        Command1: function() {
          return {
            skipHelp: true
          };
        },
        Command2: function() {
          return {
            getJson: function() {
              return {
                test2: 'bar'
              };
            }
          };
        }
      };

      var command = new HelpCommand(options);

      command.run(options, []);

      var json = convertToJson(options.ui.output);

      expect(json.commands).to.deep.equal([
        {
          test2: 'bar'
        }
      ]);
    });

    it('ignores skipHelp when single', function() {
      options.commands = {
        Command1: function() {
          return {
            skipHelp: true,
            getJson: function() {
              return {
                test1: 'bar'
              };
            }
          };
        }
      };

      var command = new HelpCommand(options);

      command.run(options, ['command-1']);

      var json = convertToJson(options.ui.output);

      expect(json.commands).to.deep.equal([
        {
          test1: 'bar'
        }
      ]);
    });

    it('lists addons', function() {
      options.project.eachAddonCommand = function(callback) {
        callback('my-addon', {
          Command1: function() {
            return {
              getJson: function() {
                return {
                  test1: 'foo'
                };
              }
            };
          },
          Command2: function() {
            return {
              getJson: function() {
                return {
                  test2: 'bar'
                };
              }
            };
          }
        });
      };

      var command = new HelpCommand(options);

      command.run(options, []);

      var json = convertToJson(options.ui.output);

      expect(json.addons).to.deep.equal([
        {
          name: 'help',
          description: 'Outputs the usage instructions for all commands or the provided command',
          aliases: [null, 'h', '--help', '-h'],
          works: 'everywhere',
          availableOptions: [
            {
              name: 'verbose',
              default: false,
              aliases: ['v'],
              key: 'verbose',
              required: false
            },
            {
              name: 'json',
              default: false,
              key: 'json',
              required: false
            }
          ],
          anonymousOptions: ['<command-name (Default: all)>'],
          commands: [
            {
              test1: 'foo'
            },
            {
              test2: 'bar'
            }
          ]
        }
      ]);
    });

    it('finds single addon command', function() {
      options.project.eachAddonCommand = function(callback) {
        callback('my-addon', {
          Command1: function() {
            return {
              getJson: function() {
                return {
                  test1: 'foo'
                };
              }
            };
          },
          Command2: function() {}
        });
      };

      var command = new HelpCommand(options);

      command.run(options, ['command-1']);

      var json = convertToJson(options.ui.output);

      expect(json.commands).to.deep.equal([
        {
          test1: 'foo'
        }
      ]);
    });
  });
});
