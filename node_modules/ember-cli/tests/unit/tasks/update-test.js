'use strict';

var fs         = require('fs');
var path       = require('path');
var expect     = require('chai').expect;
var MockUI     = require('../../helpers/mock-ui');
var Promise    = require('../../../lib/ext/promise');
var UpdateTask = require('../../../lib/tasks/update');

describe('update task', function() {
  var updateTask;
  var ui;

  var dummyPkgPath = '../../fixtures/dummy-project-outdated/package.json';

  var loadCalledWith;
  var installCalledWith;
  var initCommandWasRun;

  var npm = {
    load: function(options, callback) {
      setTimeout(function() {
        callback(undefined, npm);
      }, 0);
      loadCalledWith = options;
    },
    commands: {
      install: function(packages, callback) {
        setTimeout(callback, 0);
        installCalledWith = packages;
      }
    }
  };

  beforeEach(function() {
    installCalledWith = loadCalledWith = initCommandWasRun = undefined;
  });

  describe('don\'t update', function() {
    beforeEach(function() {
      ui = new MockUI();

      ui.prompt = function(messageObject) {
        return new Promise(function(resolve) {
          ui.write(messageObject.message);
          resolve({
            answer: false
          });
        });
      };
      updateTask = new UpdateTask({
        ui: ui,
        npm: npm
      });
    });

    it('says \'a new version is available\' and asks you to confirm you want to update', function() {
      return updateTask.run({
        environment: 'development'
      }, {
        newestVersion: '100.0.0'
      }).then(function() {
        expect(ui.output).to.include('A new version of ember-cli is available');
        expect(ui.output).to.include('Are you sure you want to update ember-cli?');
      });
    });
  });

  describe('do update', function() {
    var pkg;

    beforeEach(function() {
      ui = new MockUI();

      ui.pleasantProgress = {
        start: function() { },
        stop: function() { }
      };

      ui.prompt = function(messageObject) {
        return new Promise(function(resolve) {
          ui.write(messageObject.message);
          resolve({
            answer: true
          });
        });
      };

      function Init() {

      }

      Init.prototype.run = function() {
        initCommandWasRun = true;
      };

      updateTask = new UpdateTask({
        commands: {
          Init: Init
        },
        ui: ui,
        npm: npm,
        project: {
          root: 'tests/fixtures/dummy-project-outdated',
          pkg: require(dummyPkgPath)
        }
      });
      pkg = updateTask.project.pkg;
    });

    afterEach(function() {
      pkg.devDependencies['ember-cli'] = '0.0.1';
      fs.writeFileSync(path.join(__dirname, dummyPkgPath), JSON.stringify(pkg, null, 2));
    });

    it('says \'a new version is available\' and asks you to confirm you want to update', function() {
      this.timeout(1000000);
      return updateTask.run({
        environment: 'development'
      }, {
        newestVersion: '100.0.0'
      }).then(function() {
        expect(ui.output).to.include('A new version of ember-cli is available');
        expect(ui.output).to.include('Are you sure you want to update ember-cli?');
        expect(installCalledWith).to.deep.equal([ 'ember-cli' ], '');
        expect(loadCalledWith).to.deep.equal({
          'global': true,
          'loglevel': 'silent'
        }, '');
        expect(initCommandWasRun);
      });
    });

    it('updates package.json file with newly updated version number', function() {
      return updateTask.run({
        environment: 'development'
      }, {
        newestVersion: '100.0.0'
      }).then(function() {
        expect(pkg.devDependencies['ember-cli']).to.equal('100.0.0');
      });
    });
  });
});
