'use strict';

var path            = require('path');
var Command         = require('../models/command');
var lookupCommand   = require('../cli/lookup-command');
var string          = require('../utilities/string');
var assign          = require('lodash/object/assign');
var GenerateCommand = require('./generate');

var RootCommand = Command.extend({
  isRoot: true,
  name: 'ember',

  anonymousOptions: [
    '<command (Default: help)>'
  ]
});

module.exports = Command.extend({
  name: 'help',
  description: 'Outputs the usage instructions for all commands or the provided command',
  aliases: [undefined, 'h', '--help', '-h'],
  works: 'everywhere',

  availableOptions: [
    { name: 'verbose', type: Boolean, default: false, aliases: ['v'] },
    { name: 'json',    type: Boolean, default: false }
  ],

  anonymousOptions: [
    '<command-name (Default: all)>'
  ],

  run: function(commandOptions, rawArgs) {
    var multipleCommands = [GenerateCommand.prototype.name].concat(GenerateCommand.prototype.aliases);
    var command;
    var json;
    var rootCommand = new RootCommand({
      ui: this.ui,
      project: this.project,
      commands: this.commands,
      tasks: this.tasks
    });
    if (commandOptions.json) {
      json = rootCommand.getJson(commandOptions);
      json.commands = [];
      json.addons = [];
    }
    if (rawArgs.length === 0) {
      if (!commandOptions.json) {
        rootCommand.printBasicHelp(commandOptions);
        // Display usage for all commands.
        this.ui.writeLine('Available commands in ember-cli:');
        this.ui.writeLine('');
      }

      Object.keys(this.commands).forEach(function(commandName) {
        this._printBasicHelpForCommand(commandName, commandOptions, json);
      }, this);

      if (this.project.eachAddonCommand) {
        this.project.eachAddonCommand(function(addonName, commands) {
          this.commands = commands;
          var addonJson;
          if (commandOptions.json) {
            addonJson = this.getJson(commandOptions);
            addonJson.commands = [];
            json.addons.push(addonJson);
          } else {
            this.ui.writeLine('');
            this.ui.writeLine('Available commands from ' + addonName + ':');
          }
          Object.keys(this.commands).forEach(function(commandName) {
            this._printBasicHelpForCommand(commandName, commandOptions, addonJson);
          }, this);
        }.bind(this));
      }
    } else {
      // If args were passed to the help command,
      // attempt to look up the command for each of them.
      if (!commandOptions.json) {
        this.ui.writeLine('Requested ember-cli commands:');
        this.ui.writeLine('');
      }

      if (this.project.eachAddonCommand) {
        this.project.eachAddonCommand(function(addonName, commands) {
          assign(this.commands, commands);
        }.bind(this));
      }

      if (multipleCommands.indexOf(rawArgs[0]) > -1) {
        command = rawArgs.shift();
        if (rawArgs.length > 0) {
          commandOptions.rawArgs = rawArgs;
        }
        rawArgs = [command];
      }

      // Iterate through each arg beyond the initial 'help' command,
      // and try to display usage instructions.
      rawArgs.forEach(function(commandName) {
        this._printDetailedHelpForCommand(commandName, commandOptions, json);
      }, this);
    }

    if (commandOptions.json) {
      this._printJsonHelp(json);
    }
  },

  _printBasicHelpForCommand: function(commandName, options, json) {
    if (options.json) {
      this._addCommandHelpToJson(commandName, false, options, json);
    } else {
      this._printHelpForCommand(commandName, false, options);
    }
  },

  _printDetailedHelpForCommand: function(commandName, options, json) {
    if (options.json) {
      this._addCommandHelpToJson(commandName, true, options, json);
    } else {
      this._printHelpForCommand(commandName, true, options);
    }
  },

  _addCommandHelpToJson: function(commandName, single, options, json) {
    var command = this._lookupCommand(commandName);
    if (!command.skipHelp || single) {
      json.commands.push(command.getJson(options));
    }
  },

  _printHelpForCommand: function(commandName, detailed, options) {
    var command = this._lookupCommand(commandName);

    if (!command.skipHelp || detailed) {
      command.printBasicHelp(options);
    }

    if (detailed) {
      command.printDetailedHelp(options);
    }
  },

  _printJsonHelp: function(json) {
    var outputJsonString = JSON.stringify(json, function(key, value) {
      // build command has a recursive property
      if (value === path) {
        return 'path';
      }
      return value;
    }, 2);

    this.ui.writeLine(outputJsonString);
  },

  _lookupCommand: function(commandName) {
    var Command = this.commands[string.classify(commandName)] ||
                  lookupCommand(this.commands, commandName);

    return new Command({
      ui: this.ui,
      project: this.project,
      commands: this.commands,
      tasks: this.tasks
    });
  }
});
