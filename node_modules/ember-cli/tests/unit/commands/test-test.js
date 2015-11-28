'use strict';

var path           = require('path');
var CoreObject     = require('core-object');
var expect         = require('chai').expect;
var stub           = require('../../helpers/stub').stub;
var MockProject    = require('../../helpers/mock-project');
var commandOptions = require('../../factories/command-options');
var Promise        = require('../../../lib/ext/promise');
var Task           = require('../../../lib/models/task');
var TestCommand    = require('../../../lib/commands/test');

describe('test command', function() {
  var tasks, options, command;
  var buildRun, testRun, testServerRun;

  beforeEach(function() {
    tasks = {
      Build: Task.extend(),
      Test: Task.extend(),
      TestServer: Task.extend()
    };

    var project = new MockProject();

    project.isEmberCLIProject = function() { return true; };

    options = commandOptions({
      tasks: tasks,
      testing: true,
      project: project
    });

    stub(tasks.Test.prototype, 'run', Promise.resolve());
    stub(tasks.Build.prototype, 'run', Promise.resolve());
    stub(tasks.TestServer.prototype, 'run', Promise.resolve());

    buildRun = tasks.Build.prototype.run;
    testRun = tasks.Test.prototype.run;
    testServerRun = tasks.TestServer.prototype.run;
  });

  function buildCommand() {
    command = new TestCommand(options);
  }

  describe('default', function() {
    beforeEach(function() {
      buildCommand();
    });

    it('builds and runs test', function() {
      return command.validateAndRun([]).then(function() {
        expect(buildRun.called).to.equal(1, 'expected build task to be called once');
        expect(testRun.called).to.equal(1, 'expected test task to be called once');
      });
    });

    it('has the correct options', function() {
      return command.validateAndRun([]).then(function() {
        var buildOptions = buildRun.calledWith[0][0];
        var testOptions = testRun.calledWith[0][0];

        expect(buildOptions.environment).to.equal('test', 'has correct env');
        expect(buildOptions.outputPath, 'has outputPath');
        expect(testOptions.configFile).to.equal('./testem.json', 'has config file');
        expect(testOptions.port).to.equal(7357, 'has config file');
      });
    });

    it('passes through custom configFile option', function() {
      return command.validateAndRun(['--config-file=some-random/path.json']).then(function() {
        var testOptions = testRun.calledWith[0][0];

        expect(testOptions.configFile).to.equal('some-random/path.json');
      });
    });

    it('does not pass any port options', function() {
      return command.validateAndRun([]).then(function() {
        var testOptions = testRun.calledWith[0][0];

        expect(testOptions.port).to.equal(7357);
      });
    });

    it('passes through a custom test port option', function() {
      return command.validateAndRun(['--test-port=5679']).then(function() {
        var testOptions = testRun.calledWith[0][0];

        expect(testOptions.port).to.equal(5679);
      });
    });

    it('passes through a custom test port option of 0 to allow OS to choose open system port', function() {
      return command.validateAndRun(['--test-port=0']).then(function() {
        var testOptions = testRun.calledWith[0][0];

        expect(testOptions.port).to.equal(0);
      });
    });

    it('only passes through the port option', function() {
      return command.validateAndRun(['--port=5678']).then(function() {
        var testOptions = testRun.calledWith[0][0];

        expect(testOptions.port).to.equal(5679);
      });
    });

    it('passes both the port and the test port options', function() {
      return command.validateAndRun(['--port=5678', '--test-port=5900']).then(function() {
        var testOptions = testRun.calledWith[0][0];

        expect(testOptions.port).to.equal(5900);
      });
    });

    it('passes through custom host option', function() {
      return command.validateAndRun(['--host=greatwebsite.com']).then(function() {
        var testOptions = testRun.calledWith[0][0];

        expect(testOptions.host).to.equal('greatwebsite.com');
      });
    });

    it('passes through custom reporter option', function() {
      return command.validateAndRun(['--reporter=xunit']).then(function() {
        var testOptions = testRun.calledWith[0][0];

        expect(testOptions.reporter).to.equal('xunit');
      });
    });

    it('has the correct options when called with a build path and does not run a build task', function() {
      return command.validateAndRun(['--path=tests']).then(function() {
        expect(buildRun.called).to.equal(0, 'build task not called');
        expect(testRun.called).to.equal(1, 'test task called once');

        var testOptions = testRun.calledWith[0][0];

        expect(testOptions.outputPath).to.equal(path.resolve('tests'), 'has outputPath');
        expect(testOptions.configFile).to.equal('./testem.json', 'has config file');
        expect(testOptions.port).to.equal(7357, 'has config file');
      });
    });

    it('throws an error if the build path does not exist', function() {
      return command.validateAndRun(['--path=bad/path/to/build']).then(function() {
        expect(false, 'should have rejected the build path');
      }).catch(function(error) {
        expect(error.message).to.equal('The path bad/path/to/build does not exist. Please specify a valid build directory to test.');
      });
    });
  });

  describe('--server option', function() {
    beforeEach(function() {
      options.Builder = CoreObject.extend();
      options.Watcher = CoreObject.extend();

      buildCommand();
    });

    it('builds a watcher with verbose set to false', function() {
      return command.validateAndRun(['--server']).then(function() {
        var testOptions = testServerRun.calledWith[0][0];

        expect(testOptions.watcher.verbose, false);
      });
    });

    it('builds a watcher with options.watcher set to value provided', function() {
      return command.validateAndRun(['--server', '--watcher=polling']).then(function() {
        var testOptions = testServerRun.calledWith[0][0];

        expect(testOptions.watcher.options.watcher).to.equal('polling');
      });
    });

    it('throws an error if using a build path', function() {
      return command.validateAndRun(['--server', '--path=tests']).then(function() {
        expect(false, 'should have rejected using a build path with the server');
      }).catch(function(error) {
        expect(error.message).to.equal('Specifying a build is not allowed with the `--server` option.');
      });
    });
  });

  describe('_generateCustomConfigs', function() {
    var runOptions;

    beforeEach(function() {
      buildCommand();
      runOptions = {};
    });

    it('should return an object even if passed param is empty object', function() {
      var result = command._generateCustomConfigs(runOptions);
      expect(result).to.be.an('object');
    });

    it('when launch option is present, should be reflected in returned config', function() {
      runOptions.launch = 'fooLauncher';
      var result = command._generateCustomConfigs(runOptions);

      expect(result.launcher, 'fooLauncher');
    });

    it('when query option is present, should be reflected in returned config', function() {
      runOptions.query = 'someQuery=test';
      var result = command._generateCustomConfigs(runOptions);

      expect(result.queryString).to.equal(runOptions.query);
    });

    it('when provided test-page the new file returned contains the value in test_page', function() {
      runOptions['test-page'] = 'foo/test.html?foo';
      var result = command._generateCustomConfigs(runOptions);

      expect(result.testPage).to.be.equal('foo/test.html?foo&');
    });

    it('when provided test-page with filter, module, and query the new file returned contains those values in test_page', function() {
      runOptions.module = 'fooModule';
      runOptions.filter = 'bar';
      runOptions.query = 'someQuery=test';
      runOptions['test-page'] = 'foo/test.html?foo';
      var contents = command._generateCustomConfigs(runOptions);

      expect(contents.testPage).to.be.equal('foo/test.html?foo&module=fooModule&filter=bar&someQuery=test');
    });

    it('when provided test-page with filter and module the new file returned contains both option values in test_page', function() {
      runOptions.module = 'fooModule';
      runOptions.filter = 'bar';
      runOptions['test-page'] = 'foo/test.html?foo';
      var contents = command._generateCustomConfigs(runOptions);

      expect(contents.testPage).to.be.equal('foo/test.html?foo&module=fooModule&filter=bar');
    });

    it('when provided test-page with filter and query the new file returned contains both option values in test_page', function() {
      runOptions.query = 'someQuery=test';
      runOptions.filter = 'bar';
      runOptions['test-page'] = 'foo/test.html?foo';
      var contents = command._generateCustomConfigs(runOptions);

      expect(contents.testPage).to.be.equal('foo/test.html?foo&filter=bar&someQuery=test');
    });

    it('when provided test-page with module and query the new file returned contains both option values in test_page', function() {
      runOptions.module = 'fooModule';
      runOptions.query = 'someQuery=test';
      runOptions['test-page'] = 'foo/test.html?foo';
      var contents = command._generateCustomConfigs(runOptions);

      expect(contents.testPage).to.be.equal('foo/test.html?foo&module=fooModule&someQuery=test');
    });

    it('when provided launch the new file returned contains the value in launch', function() {
      runOptions.launch = 'fooLauncher';
      var contents = command._generateCustomConfigs(runOptions);

      expect(contents['launch']).to.be.equal('fooLauncher');
    });

    it('when provided filter is all lowercase to match the test name', function() {
      runOptions['test-page'] = 'tests/index.html';
      runOptions.filter = 'BAR';
      var contents = command._generateCustomConfigs(runOptions);

      expect(contents.testPage).to.be.equal('tests/index.html?filter=bar');
    });

    it('when module and filter option is present uses buildTestPageQueryString for test_page queryString', function() {
      runOptions.filter = 'bar';
      runOptions['test-page'] = 'tests/index.html';
      command.buildTestPageQueryString = function(options) {
        expect(options).to.deep.equal(runOptions);

        return 'blah=zorz';
      };

      var contents = command._generateCustomConfigs(runOptions);

      expect(contents.testPage).to.be.equal('tests/index.html?blah=zorz');
    });

    it('new file returned contains the filter option value in test_page', function() {
      runOptions.filter = 'foo';
      runOptions['test-page'] = 'tests/index.html';
      var contents = command._generateCustomConfigs(runOptions);

      expect(contents.testPage).to.be.equal('tests/index.html?filter=foo');
    });

    it('adds with a `&` if query string contains `?` already', function() {
      runOptions.filter = 'foo';
      runOptions['test-page'] = 'tests/index.html?hidepassed';
      var contents = command._generateCustomConfigs(runOptions);

      expect(contents.testPage).to.be.equal('tests/index.html?hidepassed&filter=foo');
    });

    it('new file returned contains the module option value in test_page', function() {
      runOptions.module = 'fooModule';
      runOptions['test-page'] = 'tests/index.html';
      var contents = command._generateCustomConfigs(runOptions);

      expect(contents.testPage).to.be.equal('tests/index.html?module=fooModule');
    });

    it('new file returned contains the query option value in test_page', function() {
      runOptions.query = 'someQuery=test';
      runOptions['test-page'] = 'tests/index.html';
      var contents = command._generateCustomConfigs(runOptions);

      expect(contents.testPage).to.be.equal('tests/index.html?someQuery=test');
    });

    it('new file returned contains the query option value with multiple queries in test_page', function() {
      runOptions.query = 'someQuery=test&something&else=false';
      runOptions['test-page'] = 'tests/index.html';
      var contents = command._generateCustomConfigs(runOptions);

      expect(contents.testPage).to.be.equal('tests/index.html?someQuery=test&something&else=false');
    });
  });
});
