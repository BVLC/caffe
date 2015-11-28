'use strict';

var chalk       = require('chalk');
var Command     = require('../models/command');
var Promise     = require('../ext/promise');
var Blueprint   = require('../models/blueprint');
var merge       = require('lodash/object/merge');
var reject      = require('lodash/collection/reject');
var EOL         = require('os').EOL;
var SilentError = require('silent-error');

module.exports = Command.extend({
  name: 'generate',
  description: 'Generates new code from blueprints.',
  aliases: ['g'],
  works: 'insideProject',

  availableOptions: [
    { name: 'dry-run',       type: Boolean, default: false, aliases: ['d'] },
    { name: 'verbose',       type: Boolean, default: false, aliases: ['v'] },
    { name: 'pod',           type: Boolean, default: false, aliases: ['p'] },
    { name: 'classic',       type: Boolean, default: false, aliases: ['c'] },
    { name: 'dummy',         type: Boolean, default: false, aliases: ['dum', 'id'] },
    { name: 'in-repo-addon', type: String,  default: null,  aliases: ['in-repo', 'ir'] }
  ],

  anonymousOptions: [
    '<blueprint>'
  ],

  beforeRun: function(rawArgs) {
    if (!rawArgs.length) {
      return;
    }
    // merge in blueprint availableOptions
    var blueprint;
    try{
      blueprint = this.lookupBlueprint(rawArgs[0]);
      this.registerOptions(blueprint);
    }
    catch(e) {
      SilentError.debugOrThrow('ember-cli/commands/generate', e);
    }
  },

  run: function(commandOptions, rawArgs) {
    var blueprintName = rawArgs[0];

    if (!blueprintName) {
      return Promise.reject(new SilentError('The `ember generate` command requires a ' +
                                            'blueprint name to be specified. ' +
                                            'For more details, use `ember help`.'));
    }
    var Task = this.tasks.GenerateFromBlueprint;
    var task = new Task({
      ui: this.ui,
      analytics: this.analytics,
      project: this.project,
      testing: this.testing,
      settings: this.settings
    });

    var taskArgs = {
      args: rawArgs
    };

    if (this.settings && this.settings.usePods && !commandOptions.classic) {
      commandOptions.pod = !commandOptions.pod;
    }

    var taskOptions = merge(taskArgs, commandOptions || {});

    if (this.project.initializeAddons) {
      this.project.initializeAddons();
    }

    return task.run(taskOptions);
  },

  lookupBlueprint: function(name) {
    return Blueprint.lookup(name, {
      paths: this.project.blueprintLookupPaths()
    });
  },

  printDetailedHelp: function(options) {
    this.ui.writeLine(this.getAllBlueprints(options));
  },

  addAdditionalJsonForHelp: function(json, options) {
    json.availableBlueprints = this.getAllBlueprints(options);
  },

  getAllBlueprints: function(options) {
    var lookupPaths   = this.project.blueprintLookupPaths();
    var blueprintList = Blueprint.list({ paths: lookupPaths });

    var output = '';

    var singleBlueprintName;
    if (options.rawArgs) {
      singleBlueprintName = options.rawArgs[0];
    }

    if (!singleBlueprintName && !options.json) {
      output += EOL + '  Available blueprints:' + EOL;
    }

    var collectionsJson = [];

    blueprintList.forEach(function(collection) {
      var result = this.getPackageBlueprints(collection, options, singleBlueprintName);
      if (options.json) {
        var collectionJson = {};
        collectionJson[collection.source] = result;
        collectionsJson.push(collectionJson);
      } else {
        output += result;
      }
    }, this);

    if (singleBlueprintName && !output && !options.json) {
      output = chalk.yellow('The \'' + singleBlueprintName +
        '\' blueprint does not exist in this project.') + EOL;
    }

    if (options.json) {
      return collectionsJson;
    } else {
      return output;
    }
  },

  getPackageBlueprints: function(collection, options, singleBlueprintName) {
    var verbose    = options.verbose;
    var blueprints = collection.blueprints;

    if (!verbose) {
      blueprints = reject(blueprints, 'overridden');
    }

    var output = '';

    if (blueprints.length && !singleBlueprintName && !options.json) {
      output += '    ' + collection.source + ':' + EOL;
    }

    var blueprintsJson = [];

    blueprints.forEach(function(blueprint) {
      var singleMatch = singleBlueprintName === blueprint.name;
      if (singleMatch) {
        verbose = true;
      }
      if (!singleBlueprintName || singleMatch) {
        // this may add default keys for printing
        blueprint.availableOptions.forEach(this.normalizeOption);

        if (options.json) {
          blueprintsJson.push(blueprint.getJson(verbose));
        } else {
          output += blueprint.printBasicHelp(verbose) + EOL;
        }
      }
    }, this);

    if (options.json) {
      return blueprintsJson;
    } else {
      return output;
    }
  }
});
