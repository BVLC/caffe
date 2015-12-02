
var pmx = require('..');
var should = require('should');
var pkg = require('../package.json');

function forkWithoutEnv() {
  var app = require('child_process').fork(__dirname + '/fixtures/module/module.fixture.js', [], {
    env : {
    }
  });
  return app;
}

function forkWithSpecificVar() {
  var app = require('child_process').fork(__dirname + '/fixtures/module/module.fixture.js', [], {
    env : {
      'module' : '{ "option1" : "value1", "option2" : "value2", "initial" : "over", "password" : "1234s" }'
    }
  });
  return app;
}

describe('PMX module', function() {
  var app;
  var action_name;

  it('should emit a new action', function(done) {
    // 1 - It should emit an action
    app = forkWithoutEnv();

    function processMsg(dt) {
      if (dt.type != 'axm:option:configuration') return;
      /**
       * Right event sent
       */
      dt.type.should.eql('axm:option:configuration');

      /**
       * Options set
       */
      dt.data.description.should.eql('comment');
      dt.data.module_version.should.eql('1.0.0');
      should(dt.data.pmx_version).eql(null);
      dt.data.module_name.should.eql('module');
      dt.data.alert_enabled.should.be.true;

      /**
       * Configuration succesfully passed
       */
      dt.data.initial.should.eql('init-val');

      /**
       * Should configuration variable be mirrored into module_conf
       * attribute (for keymetrics purposes)
       */
      dt.data.module_conf.initial.should.eql('init-val');
      app.kill();
      app.removeListener('message', processMsg);
      done();
    }

    app.on('message', processMsg);
  });

  it('should emit a new action', function(done) {
    // 1 - It should emit an action
    app = forkWithSpecificVar();

    function processMsg(dt) {

      /**
       * Right event sent
       */
      dt.type.should.eql('axm:option:configuration');

      /**
       * Options set
       */
      dt.data.description.should.eql('comment');
      dt.data.module_version.should.eql('1.0.0');
      dt.data.module_name.should.eql('module');

      /**
       * Configuration succesfully passed
       */
      dt.data.option1.should.eql('value1');
      dt.data.option2.should.eql('value2');
      dt.data.initial.should.eql('over');

      /**
       * Should configuration variable be mirrored into module_conf
       * attribute (for keymetrics purposes)
       */
      dt.data.module_conf.option1.should.eql('value1');
      dt.data.module_conf.option2.should.eql('value2');
      dt.data.module_conf.initial.should.eql('over');
      app.kill();
      app.removeListener('message', processMsg);
      done();
    }

    app.on('message', processMsg);
  });

  it('should hide password', function(done) {
    app = forkWithSpecificVar();

    function processMsg(dt) {
      dt.data.alert_enabled.should.be.true;
      dt.data.password.should.eql('1234s');
      dt.data.module_conf.password.should.eql('Password hidden');
      done();
      app.removeListener('message', processMsg);
      app.kill();
    }

    app.on('message', processMsg);
  });

  it('should find existing file', function(done) {
    var content = pmx.resolvePidPaths([
      'asdasdsad',
      'asdasd',
      'lolilol',
      __dirname + '/fixtures/file.pid'
    ]);

    content.should.eql(1456);
    done();
  });

  it('should return null', function(done) {
    var content = pmx.resolvePidPaths([
      'asdasdsad',
      'asdasd',
      'lolilol'
    ]);

    should(content).be.null;
    done();
  });

});
