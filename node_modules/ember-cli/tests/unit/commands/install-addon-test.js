'use strict';

var expect              = require('chai').expect;
var MockProject         = require('../../helpers/mock-project');
var commandOptions      = require('../../factories/command-options');
var Promise             = require('../../../lib/ext/promise');
var Task                = require('../../../lib/models/task');
var AddonInstall        = require('../../../lib/tasks/addon-install');
var InstallAddonCommand = require('../../../lib/commands/install-addon');

describe('install:addon command', function() {
  var npmInstance, generateBlueprintInstance;
  var command;

  beforeEach(function() {
    var tasks = {
      AddonInstall: AddonInstall,
      NpmInstall: Task.extend({
        init: function() {
          npmInstance = this;
        },
        run: function() {
          return Promise.resolve();
        }
      }),

      GenerateFromBlueprint: Task.extend({
        init: function() {
          generateBlueprintInstance = this;
        },
        run: function() {
          return Promise.resolve();
        }
      })
    };

    var project = new MockProject();

    project.isEmberCLIProject = function() { return true; };
    project.initializeAddons  = function() { };
    project.reloadAddons      = function() {
      this.addons = [{
        pkg: {
          name: 'ember-cli-photoswipe',
          'ember-addon': {
            defaultBlueprint: 'photoswipe'
          }
        }
      }];
    };

    var options = commandOptions({
      project: project,
      tasks: tasks
    });

    command = new InstallAddonCommand(options);
  });

  it('will show a deprecation warning', function() {
    return command.validateAndRun(['ember-cli-photoswipe']).then(function() {
      expect(command.ui.output).to.include(
          'This command has been deprecated. Please use `ember install ' +
          '<addonName>` instead.');

      expect(npmInstance.ui, 'ui was set');
      expect(npmInstance.project, 'project was set');
      expect(npmInstance.analytics, 'analytics was set');

      expect(generateBlueprintInstance.ui, 'ui was set');
      expect(generateBlueprintInstance.project, 'project was set');
      expect(generateBlueprintInstance.analytics, 'analytics was set');
    });
  });
});
