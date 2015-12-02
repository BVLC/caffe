
var pmx    = require('..');
var should = require('should');
var Plan   = require('./helpers/plan.js');

function forkAlertedModule() {
  var app = require('child_process').fork(__dirname + '/fixtures/module/module.fixture.js', [], {
    env : {
    }
  });
  return app;
}

function forkAlertedModuleThresholdAvg() {
  var app = require('child_process').fork(__dirname + '/fixtures/module/module-avg.fixture.js', [], {
    env : {
    }
  });
  return app;
}

function forkNonAlertedModule() {
  var app = require('child_process').fork(__dirname + '/fixtures/module/module.alert-off.fixture.js', [], {
    env : {
    }
  });
  return app;
}

describe('Alert system', function() {
  var app;

  describe('(MODULE) With Alert', function() {
    it('should start module with alert activated', function(done) {
      app = forkAlertedModule();

      function processMsg(dt) {
        if (!dt.data.alert_enabled) return;
        dt.data.alert_enabled.should.be.true;
        app.removeListener('message', processMsg);
        done();
      }

      app.on('message', processMsg);
    });

    it('should receive notification threshold alert', function(done) {
      var plan = new Plan(2, function() {
        app.kill();
        app.removeListener('message', processMsg);
        done();
      });

      function processMsg(dt) {
        if (dt.type == 'axm:monitor') {
          dt.data['probe-test'].alert.value.should.eql(15);
          dt.data['probe-test'].alert.mode.should.eql('threshold');
          dt.data['probe-test'].alert.cmp.should.eql('>');
          plan.ok(true);
        }

        if (dt.type == 'process:exception') {
          should(dt.data.message).startWith('val too high');
          plan.ok(true);
        }
      }

      app.on('message', processMsg);
    });
  });

  describe('(MODULE) Alert threshold', function() {
    it('should launch app', function(done) {
      app = forkAlertedModuleThresholdAvg();

      function processMsg(dt) {
        if (!dt.data.alert_enabled) return;
        dt.data.alert_enabled.should.be.true;
        app.removeListener('message', processMsg);
        done();
      }

      app.on('message', processMsg);
    });

    it('should receive notification threshold alert', function(done) {
      var plan = new Plan(2, function() {
        app.kill();
        app.removeListener('message', processMsg);
        done();
      });

      function processMsg(dt) {
        if (dt.type == 'axm:monitor') {
          dt.data['probe-test'].alert.value.should.eql(15);
          dt.data['probe-test'].alert.interval.should.eql(5);
          dt.data['probe-test'].alert.mode.should.eql('threshold-avg');
          dt.data['probe-test'].alert.cmp.should.eql('>');
          plan.ok(true);
        }

        if (dt.type == 'process:exception') {
          should(dt.data.message).be.equal('val too high');
          plan.ok(true);
        }
      }

      app.on('message', processMsg);
    });
  });

  describe.skip('(MODULE) Without Alert', function() {
    it('should start module with alert activated', function(done) {
      app = forkNonAlertedModule();

      app.once('message', function(dt) {
        dt.data.alert_enabled.should.be.false;
        done();
      });

    });

    it('should not receive notification threshold alert', function(done) {
      function processMsg(dt) {
        if (dt.type == 'axm:monitor') {
          // No alert but alert field should exist
          dt.data['probe-test'].alert.should.exists;
          Object.keys(dt.data['probe-test'].alert).length.should.eql(0);
        }

        if (dt.type == 'process:exception') {
          done('ERROR EMITTED :/');
        }
      }

      console.log('Waiting 3secs (no alert should be emitted)');

      setTimeout(function() {
        app.removeListener('message', processMsg);
        app.kill();
        done();
      }, 3000);
      app.on('message', processMsg);
    });

  });


});
