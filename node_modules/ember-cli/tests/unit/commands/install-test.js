'use strict';

var expect         = require('chai').expect;
var stub           = require('../../helpers/stub').stub;
var MockProject    = require('../../helpers/mock-project');
var commandOptions = require('../../factories/command-options');
var Task           = require('../../../lib/models/task');
var Promise        = require('../../../lib/ext/promise');
var AddonInstall   = require('../../../lib/tasks/addon-install');
var InstallCommand = require('../../../lib/commands/install');

describe('install command', function() {
  var generateBlueprintInstance, npmInstance;
  var command, tasks;

  beforeEach(function() {
    var project = new MockProject();

    project.isEmberCLIProject = function() {
      return true;
    };

    project.initializeAddons = function() {};
    project.reloadAddons = function() {
      this.addons = [
        {
          pkg: {
            name: 'ember-data',
          }
        },
        {
          pkg: {
            name: 'ember-cli-cordova',
            'ember-addon': {
              defaultBlueprint: 'cordova-starter-kit'
            }
          }
        },
        {
          pkg: {
            name: 'ember-cli-qunit'
          }
        }
      ];
    };

    tasks = {
      AddonInstall: AddonInstall,
      NpmInstall: Task.extend({
        project: project,
        init: function() {
          npmInstance = this;
        }
      }),

      GenerateFromBlueprint: Task.extend({
        project: project,
        init: function() {
          generateBlueprintInstance = this;
        }
      })
    };

    var options = commandOptions({
      project: project,
      tasks: tasks
    });

    stub(tasks.NpmInstall.prototype, 'run', Promise.resolve());
    stub(tasks.GenerateFromBlueprint.prototype, 'run', Promise.resolve());

    command = new InstallCommand(options);
  });

  afterEach(function() {
    tasks.NpmInstall.prototype.run.restore();
    tasks.GenerateFromBlueprint.prototype.run.restore();
  });

  it('initializes npm install and generate blueprint task with ui, project and analytics', function() {
    return command.validateAndRun(['ember-data']).then(function() {
      expect(npmInstance.ui, 'ui was set');
      expect(npmInstance.project, 'project was set');
      expect(npmInstance.analytics, 'analytics was set');

      expect(generateBlueprintInstance.ui, 'ui was set');
      expect(generateBlueprintInstance.project, 'project was set');
      expect(generateBlueprintInstance.analytics, 'analytics was set');
    });
  });

  describe('with args', function() {
    it('runs the npm install task with given name and save-dev true', function() {
      return command.validateAndRun(['ember-data']).then(function() {
        var npmRun = tasks.NpmInstall.prototype.run;
        expect(npmRun.called).to.equal(1, 'expected npm install run was called once');

        expect(npmRun.calledWith[0][0]).to.deep.equal({
          packages: ['ember-data'],
          'save-dev': true,
          'save-exact': true
        }, 'expected npm install called with given name and save-dev true');
      });
    });

    it('runs the package name blueprint task with given name and args', function() {
      return command.validateAndRun(['ember-data']).then(function() {
        var generateRun = tasks.GenerateFromBlueprint.prototype.run;
        expect(generateRun.calledWith[0][0].ignoreMissingMain, true);
        expect(generateRun.calledWith[0][0].args).to.deep.equal([
          'ember-data'
        ], 'expected generate blueprint called with correct args');
      });
    });

    it('fails to install second argument for unknown addon', function() {
      return command.validateAndRun(['ember-cli-cordova', 'com.ember.test']).then(function() {
        expect(false, 'should reject with error');
      }).catch(function(error) {
        var generateRun = tasks.GenerateFromBlueprint.prototype.run;
        expect(generateRun.calledWith[0][0].ignoreMissingMain, true);
        expect(generateRun.calledWith[0][0].args).to.deep.equal([
          'cordova-starter-kit'
        ], 'expected generate blueprint called with correct args');
        expect(error.message).to.equal(
          'Install failed. Could not find addon with name: com.ember.test',
          'expected error to have helpful message'
        );
      });
    });

    it('runs npmInstall once and installs three addons', function() {
      return command.validateAndRun([
        'ember-data', 'ember-cli-cordova', 'ember-cli-qunit'
      ]).then(function() {
        var npmRun = tasks.NpmInstall.prototype.run;

        expect(npmRun.called).to.equal(1, 'expect npm install to be called once');

        expect(npmRun.calledWith[0][0]).to.deep.equal({
          packages: ['ember-data', 'ember-cli-cordova', 'ember-cli-qunit'],
          'save-dev': true,
          'save-exact': true
        }, 'expected npm install called with given name and save-dev true');

        var generateRun = tasks.GenerateFromBlueprint.prototype.run;
        expect(generateRun.called).to.equal(3, 'expect blueprint generator to run thrice.');
        expect(generateRun.calledWith[0][0].args[0]).to.equal('ember-data');
        expect(generateRun.calledWith[1][0].args[0]).to.equal('cordova-starter-kit');
        expect(generateRun.calledWith[2][0].args[0]).to.equal('ember-cli-qunit');
      });
    });

    it('runs the package name blueprint task when given github/name and args', function() {
      return command.validateAndRun(['ember-cli/ember-cli-qunit']).then(function() {
        var generateRun = tasks.GenerateFromBlueprint.prototype.run;
        expect(generateRun.calledWith[0][0].ignoreMissingMain, true);
        expect(generateRun.calledWith[0][0].args).to.deep.equal([
          'ember-cli-qunit'
        ], 'expected generate blueprint called with correct args');
      });
    });

    it('gives helpful message if it can\'t find the addon', function() {
      return command.validateAndRun(['unknown-addon']).then(function() {
        expect(false, 'should reject with error');
      }).catch(function(error) {
        expect(error.message).to.equal(
          'Install failed. Could not find addon with name: unknown-addon',
          'expected error to have helpful message'
        );
      });
    });
  });

  describe('without args', function() {
    it('gives a helpful message if no arguments are passed', function() {
      return command.validateAndRun([]).then(function() {
        expect(false, 'should reject with error');
      }).catch(function(error) {
        expect(error.message).to.equal(
          'The `install` command must take an argument with the name ' +
          'of an ember-cli addon. For installing all npm and bower ' +
          'dependencies you can run `npm install && bower install`.',
          'expect error to have a helpful message'
        );
      });
    });
  });
});
