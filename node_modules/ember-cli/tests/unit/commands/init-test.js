'use strict';

var fs            = require('fs');
var os            = require('os');
var path          = require('path');
var expect        = require('chai').expect;
var MockUI        = require('../../helpers/mock-ui');
var MockAnalytics = require('../../helpers/mock-analytics');
var Promise       = require('../../../lib/ext/promise');
var Project       = require('../../../lib/models/project');
var Task          = require('../../../lib/models/task');
var InitCommand   = require('../../../lib/commands/init');

describe('init command', function() {
  var ui, analytics, tasks, command;

  beforeEach(function() {
    ui = new MockUI();
    analytics = new MockAnalytics();
    tasks = {
      InstallBlueprint: Task.extend({}),
      NpmInstall: Task.extend({}),
      BowerInstall: Task.extend({})
    };
  });

  function buildCommand(projectOpts) {
    var options = {
      ui: ui,
      analytics: analytics,
      project: new Project(process.cwd(), projectOpts || { name: 'some-random-name' }),
      tasks: tasks,
      settings: {}
    };

    command = new InitCommand(options);
  }

  it('doesn\'t allow to create an application named `test`', function() {
    buildCommand({ name: 'test' });

    return command.validateAndRun([]).then(function() {
      expect(false, 'should have rejected with an application name of test');
    })
    .catch(function(error) {
      expect(error.message).to.equal('We currently do not support a name of `test`.');
    });
  });

  it('doesn\'t allow to create an application without project name', function() {
    buildCommand({ name: undefined });

    return command.validateAndRun([]).then(function() {
      expect(false, 'should have rejected with an application without project name');
    })
    .catch(function(error) {
      expect(error.message).to.equal('The `ember init` command requires a package.json in current folder with name attribute or a specified name via arguments. For more details, use `ember help`.');
    });
  });

  it('Uses the name of the closest project to when calling installBlueprint', function() {
    tasks.InstallBlueprint = Task.extend({
      run: function(blueprintOpts) {
        expect(blueprintOpts.rawName).to.equal('some-random-name');
        return Promise.reject('Called run');
      }
    });

    buildCommand();

    return command.validateAndRun([])
      .catch(function(reason) {
        expect(reason).to.equal('Called run');
      });
  });

  it('Uses the provided app name over the closest found project', function() {
    tasks.InstallBlueprint = Task.extend({
      run: function(blueprintOpts) {
        expect(blueprintOpts.rawName).to.equal('provided-name');
        return Promise.reject('Called run');
      }
    });

    buildCommand();

    return command.validateAndRun(['--name=provided-name'])
      .catch(function(reason) {
        expect(reason).to.equal('Called run');
      });
  });


  it('Uses process.cwd if no package is found when calling installBlueprint', function() {
    // change the working dir so `process.cwd` can't be a invalid path for base directories
    // named `ember-cli`.

    var tmpDir = os.tmpdir();
    var workingDir = tmpDir + '/ember-cli-test-project';
    var currentWorkingDir = process.cwd();

    fs.mkdirSync(workingDir);
    process.chdir(workingDir);

    tasks.InstallBlueprint = Task.extend({
      run: function(blueprintOpts) {
        expect(blueprintOpts.rawName).to.equal(path.basename(process.cwd()));
        return Promise.reject('Called run');
      }
    });

    buildCommand({ name: path.basename(process.cwd()) });

    return command.validateAndRun([])
      .catch(function(reason) {
        expect(reason).to.equal('Called run');
      })
      .then(function() {
        process.chdir(currentWorkingDir);
        fs.rmdirSync(workingDir);
      });
  });

  it('doesn\'t use --dry-run or any other command option as the name', function() {
    tasks.InstallBlueprint = Task.extend({
      run: function(blueprintOpts) {
        expect(blueprintOpts.rawName).to.equal('some-random-name');
        return Promise.reject('Called run');
      }
    });

    buildCommand();

    return command.validateAndRun(['--dry-run'])
      .catch(function(reason) {
        expect(reason).to.equal('Called run');
      });
  });

  it('doesn\'t use . as the name', function() {
    tasks.InstallBlueprint = Task.extend({
      run: function(blueprintOpts) {
        expect(blueprintOpts.rawName).to.equal('some-random-name');
        return Promise.reject('Called run');
      }
    });

    buildCommand();

    return command.validateAndRun(['.'])
      .catch(function(reason) {
        expect(reason).to.equal('Called run');
      });
  });

  it('Uses the "app" blueprint by default', function() {
    tasks.InstallBlueprint = Task.extend({
      run: function(blueprintOpts) {
        expect(blueprintOpts.blueprint).to.equal('app');
        return Promise.reject('Called run');
      }
    });

    buildCommand();

    return command.validateAndRun(['--name=provided-name'])
      .catch(function(reason) {
        expect(reason).to.equal('Called run');
      });
  });

  it('Uses arguments to select files to init', function() {
    tasks.InstallBlueprint = Task.extend({
      run: function(blueprintOpts) {
        expect(blueprintOpts.blueprint).to.equal('app');
        return Promise.reject('Called run');
      }
    });

    buildCommand();

    return command.validateAndRun(['package.json', '--name=provided-name'])
      .catch(function(reason) {
        expect(reason).to.equal('Called run');
      });
  });

  it('Uses the "addon" blueprint for addons', function() {
    tasks.InstallBlueprint = Task.extend({
      run: function(blueprintOpts) {
        expect(blueprintOpts.blueprint).to.equal('addon');
        return Promise.reject('Called run');
      }
    });

    buildCommand({ keywords: ['ember-addon'], name: 'some-random-name' });

    return command.validateAndRun(['--name=provided-name'])
      .catch(function(reason) {
        expect(reason).to.equal('Called run');
      });
  });
});
