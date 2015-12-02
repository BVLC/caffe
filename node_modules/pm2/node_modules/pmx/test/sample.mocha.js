process.env.DEBUG="axm:alert:checker";

var Sample      = require('../lib/utils/sample.js');
var Plan        = require('./helpers/plan.js');

describe('Sample Checker', function() {
  it('should work with small sampling', function(done) {
    var s2 = new Sample();
    for (var i = 0; i <= 50; i++) {
      s2.update(i);    
    }
    if (s2._count === i && s2.getMean() === 25)
      done();
    else
      done(new Error('Sample: incorrect small sampling'))
  });
  it('should not go over [size] samples', function(done) {
    var s1 = new Sample(100);
    for (var i = 0; i <= 120; i++) { 
      s1.update(i);
    }
    if (s1._count === 100 && s1.getMean() === 70.5)
      done();
    else
      done(new Error('Sample: incorrect sampling'));
  });
})
