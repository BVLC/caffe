'use strict';
/*jshint expr: true*/

var expect        = require('chai').expect;
var lookupCommand = require('../../../lib/cli/lookup-command');
var Command       = require('../../../lib/models/command');
var Project       = require('../../../lib/models/project');
var MockUI        = require('../../helpers/mock-ui');
var AddonCommand  = require('../../fixtures/addon/commands/addon-command');
var OtherCommand  = require('../../fixtures/addon/commands/other-addon-command');
var ClassCommand  = require('../../fixtures/addon/commands/addon-command-class');
var OverrideCommand  = require('../../fixtures/addon/commands/addon-override-intentional');

var commands = {
  serve: Command.extend({
    name: 'serve',
    aliases: ['s'],
    works: 'everywhere',
    availableOptions: [
      { name: 'port', key: 'port', type: Number, default: 4200, required: true }
    ],
    run: function() {}
  })
};

function AddonServeCommand() { return this; }
AddonServeCommand.prototype.includedCommands = function() {
  return {
    'Serve': {
      name: 'serve',
      description: 'overrides the serve command'
    }
  };
};

describe('cli/lookup-command.js', function() {
  var ui;
  var project = {
    isEmberCLIProject: function(){ return true; },
    initializeAddons: function() {
      this.addons = [new AddonCommand(), new OtherCommand(), new ClassCommand()];
    },
    addonCommands: Project.prototype.addonCommands,
    eachAddonCommand: Project.prototype.eachAddonCommand
  };

  before(function(){
    ui = new MockUI();
  });

  it('lookupCommand() should find commands by name and aliases.', function() {
    // Valid commands

    expect(lookupCommand(commands, 'serve')).to.exist;
    expect(lookupCommand(commands, 's')).to.exist;
  });

  it('lookupCommand() should find commands that addons add by name and aliases.', function() {
    var command, Command;

    Command = lookupCommand(commands, 'addon-command', [], {
      project: project,
      ui: ui
    });
    command = new Command({
      ui: ui,
      project: project
    });

    expect(command.name).to.equal('addon-command');

    Command = lookupCommand(commands, 'ac', [], {
      project: project,
      ui: ui
    });

    command = new Command({
      ui: ui,
      project: project
    });

    expect(command.name).to.equal('addon-command');

    Command = lookupCommand(commands, 'other-addon-command', [], {
      project: project,
      ui: ui
    });

    command = new Command({
      ui: ui,
      project: project
    });

    expect(command.name).to.equal('other-addon-command');

    Command = lookupCommand(commands, 'oac', [], {
      project: project,
      ui: ui
    });

    command = new Command({
      ui: ui,
      project: project
    });

    expect(command.name).to.equal('other-addon-command');

    Command = lookupCommand(commands, 'class-addon-command', [], {
      project: project,
      ui: ui
    });

    command = new Command({
      ui: ui,
      project: project
    });

    expect(command.name).to.equal('class-addon-command');

  });

  it('lookupCommand() should write out a warning when overriding a core command', function() {
    project = {
      isEmberCLIProject: function(){ return true; },
      initializeAddons: function() {
        this.addons = [new AddonServeCommand()];
      },
      addonCommands: Project.prototype.addonCommands,
      eachAddonCommand: Project.prototype.eachAddonCommand
    };

    lookupCommand(commands, 'serve', [], {
      project: project,
      ui: ui
    });

    expect(ui.output).to.match(/warning: An ember-addon has attempted to override the core command "serve"\. The core command will be used.*/);
  });

  it('lookupCommand() should write out a warning when overriding a core command and allow it if intentional', function() {
    project = {
      isEmberCLIProject: function(){ return true; },
      initializeAddons: function() {
        this.addons = [new OverrideCommand()];
      },
      addonCommands: Project.prototype.addonCommands,
      eachAddonCommand: Project.prototype.eachAddonCommand
    };

    lookupCommand(commands, 'serve', [], {
      project: project,
      ui: ui
    });

    expect(ui.output).to.match(/warning: An ember-addon has attempted to override the core command "serve"\. The addon command will be used as the overridding was explicit.*/);
  });

  it('lookupCommand() should return UnknownCommand object when command name is not present.', function() {
    var Command = lookupCommand(commands, 'something-else', [], {
      project: project,
      ui: ui
    });
    var command = new Command({
      ui: ui,
      project: project
    });
    expect(function() {
      command.validateAndRun([]);
    }).to.throw(/command.*something-else.*is invalid/);
  });
});
