
var axm = require('..');

function fork() {
  return require('child_process').fork(__dirname + '/fixtures/event.mock.js', []);
}

describe('Event', function() {
  it('should have right property', function(done) {
    axm.should.have.property('emit');
    done();
  });

  describe('Event scenario', function() {
    var app;

    before(function() {
      app = fork();
    });

    after(function() {
      process.kill(app.pid);
    });

    it('should send right event data when called', function(done) {
      var Plan = require('./helpers/plan');
      var plan = new Plan(3, done);

      app.on('message', function(data) {

        if (data.data.__name == 'is object') {
          data.type.should.eql('human:event');
          data.data.user.should.eql('toto');
          data.data.subobj.subobj.a.should.eql('b');
          plan.ok(true);
        }

        if (data.data.__name == 'is string') {
          data.type.should.eql('human:event');
          data.data.data.should.eql('HEY!');
          plan.ok(true);
        }

        if (data.data.__name == 'is bool') {
          data.type.should.eql('human:event');
          data.data.data.should.eql(true);
          plan.ok(true);
        }

      });
    });
  });



});
