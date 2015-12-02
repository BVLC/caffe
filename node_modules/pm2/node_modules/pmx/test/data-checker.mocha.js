/*
process.env.DEBUG="axm:smart:checker";

var dataChecker = require('../lib/utils/datachecker.js');
var Plan        = require('./helpers/plan.js');

describe('Smart Data Checker', function() {
  var current_value = 10;

  it('Scenario 0 - handle 0 and linear augmentation', function(done) {

    var dtCheck = new dataChecker({
      timer : 100,
      dev   : 1,
      callback : function() {
        done(new Error('should not be called'));
      },
      refresh : function() {
        return current_value;
      }
    });

    var interval = setInterval(function() {
      current_value++;
    }, 100);


    setTimeout(function() {
      clearInterval(interval);
      dtCheck.stop();

      // Success!
      done();
    }, 15000);
  });

  it('Scenario 1 - slow increment with short anomalies', function(done) {

    var dtCheck = new dataChecker({
      timer : 100,
      dev   : 1,
      callback : function() {
        done(new Error('should not be called'));
      },
      refresh : function() {
        return current_value;
      }
    });

    var interval = setInterval(function() {
      current_value++;

      if (current_value % 30 == 0)
        current_value += 500;
      else if (current_value > 500)
        current_value -= 500;
    }, 100);


    setTimeout(function() {
      clearInterval(interval);
      dtCheck.stop();

      // Success!
      done();
    }, 15000);
  });

  it.skip('Scenario 1 - slow increment with short anomalies', function(done) {

    var dtCheck = new dataChecker({
      timer : 100,
      callback : function() {
        done(new Error('should not be called'));
      },
      refresh : function() {
        return current_value;
      }
    });

    setInterval(function() {
      current_value++;
      
      if (current_value % 30 == 0)
        current_value += 500;
      else if (current_value > 500)
        current_value -= 500;
    }, 100);

    setTimeout(function() {
      // Success!
      done();
    }, 15000);
  });

 it('Scenario 2 - calculated with mean over 30s', function(done) {

    var dtCheck = new dataChecker({
      callback : function() {
        done(new Error('should not be called'));
      },
      refresh : function() {
        return current_value;
      },
      timer : 100,
      launch: 30000,
      dev   : 0.2,
      ceil  : 1,
      calcDev: 'ema'
    });

    var interval = setInterval(function() {
      current_value++;

      if (current_value % 30 == 0)
        current_value += 500;
      else if (current_value > 500)
        current_value -= 500;
    }, 100);


    setTimeout(function() {
      clearInterval(interval);
      dtCheck.stop();

      // Success!
      done();
    }, 15000);
  });


});*/
