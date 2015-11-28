'use strict';

var expect         = require('chai').expect;
var EOL            = require('os').EOL;
var proxyquire     = require('proxyquire');
var stub           = require('../../helpers/stub').stub;
var commandOptions = require('../../factories/command-options');
var Task           = require('../../../lib/models/task');

var getPortStub;

var ServeCommand = proxyquire('../../../lib/commands/serve', {
  'portfinder': {
    getPort: function() {
      return getPortStub.apply(this, arguments);
    }
  }
});

describe('serve command', function() {
  var tasks, options, command;

  beforeEach(function() {
    tasks = {
      Serve: Task.extend()
    };

    options = commandOptions({
      tasks: tasks
    });

    stub(tasks.Serve.prototype, 'run');
    getPortStub = function(options, callback) {
      callback(null, 49152);
    };

    command = new ServeCommand(options);
  });

  afterEach(function() {
    tasks.Serve.prototype.run.restore();
  });

  it('has correct options', function() {
    return command.validateAndRun([
      '--port', '4000'
    ]).then(function() {
      var serveRun = tasks.Serve.prototype.run;
      var runOps = serveRun.calledWith[0][0];

      expect(serveRun.called).to.equal(1, 'expected run to be called once');

      expect(runOps.port).to.equal(4000, 'has correct port');
      expect(runOps.liveReloadPort).to.be.within(49152, 65535, 'has correct liveReload port');
    });
  });

  it('allows OS to choose port', function() {
    return command.validateAndRun([
      '--port', '0'
    ]).then(function() {
      var serveRun = tasks.Serve.prototype.run;
      var runOps = serveRun.calledWith[0][0];

      expect(serveRun.called).to.equal(1, 'expected run to be called once');

      expect(runOps.port).to.be.within(49152, 65535, 'has correct port');
    });
  });

  it('has correct liveLoadPort', function() {
    return command.validateAndRun([
      '--live-reload-port', '4001'
    ]).then(function() {
      var serveRun = tasks.Serve.prototype.run;
      var ops = serveRun.calledWith[0][0];

      expect(serveRun.called).to.equal(1, 'expected run to be called once');

      expect(ops.liveReloadPort).to.equal(4001, 'has correct liveReload port');
    });
  });

  it('has correct liveLoadHost', function() {
    var getPortOpts;
    getPortStub = function(options, callback) {
      getPortOpts = options;
      callback(null, 49152);
    };

    return command.validateAndRun([
      '--live-reload-host', '127.0.0.1'
    ]).then(function() {
      var serveRun = tasks.Serve.prototype.run;
      var runOps = serveRun.calledWith[0][0];

      expect(serveRun.called).to.equal(1, 'expected run to be called once');

      expect(getPortOpts.host).to.equal('127.0.0.1', 'gets a port based on the liveReload host');
      expect(runOps.liveReloadHost).to.equal('127.0.0.1', 'has correct liveReload host');
    });
  });

  it('has correct proxy', function() {
    return command.validateAndRun([
      '--proxy', 'http://localhost:3000/'
    ]).then(function() {
      var serveRun = tasks.Serve.prototype.run;
      var ops = serveRun.calledWith[0][0];

      expect(serveRun.called).to.equal(1, 'expected run to be called once');

      expect(ops.proxy).to.equal('http://localhost:3000/', 'has correct port');
    });
  });

  it('has correct insecure proxy option', function() {
    return command.validateAndRun([
      '--insecure-proxy'
    ]).then(function() {
      var serveRun = tasks.Serve.prototype.run;
      var ops = serveRun.calledWith[0][0];

      expect(serveRun.called).to.equal(1, 'expected run to be called once');

      expect(ops.insecureProxy).to.equal(true, 'has correct insecure proxy option');
    });
  });

  it('has correct default value for insecure proxy', function() {
    return command.validateAndRun().then(function() {
      var serveRun = tasks.Serve.prototype.run;
      var ops = serveRun.calledWith[0][0];

      expect(serveRun.called).to.equal(1, 'expected run to be called once');

      expect(ops.insecureProxy).to.equal(false, 'has correct insecure proxy option when not set');
    });
  });

  it('requires proxy URL to include protocol', function() {
    return command.validateAndRun([
      '--proxy', 'localhost:3000'
    ]).then(function() {
      expect(false, 'it rejects when proxy URL doesn\'t include protocol');
    })
    .catch(function(error) {
      expect(error.message).to.equal(
        'You need to include a protocol with the proxy URL.' + EOL + 'Try --proxy http://localhost:3000'
      );
    });
  });

  it('uses baseURL of correct environment', function() {
    options.project.config = function(env) {
      return { baseURL: env };
    };

    return command.validateAndRun([
      '--environment', 'test'
    ]).then(function() {
      var serveRun = tasks.Serve.prototype.run;
      var ops = serveRun.calledWith[0][0];

      expect(ops.baseURL).to.equal('test', 'Uses the correct environment.');
    });
  });

  it('host alias does not conflict with help alias', function() {
    return command.validateAndRun([
      '-H', 'hostname'
    ]).then(function() {
      var serveRun = tasks.Serve.prototype.run;
      var ops = serveRun.calledWith[0][0];

      expect(serveRun.called).to.equal(1, 'expected run to be called once');
      expect(ops.host).to.equal('hostname', 'has correct hostname');
    });
  });
});
