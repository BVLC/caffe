process.env.DEBUG="axm:alert:checker";

var Alert       = require('../lib/utils/alert.js');
var Plan        = require('./helpers/plan.js');

describe('Alert Probe Checker', function() {
  var current_value = 10;

  it('should detect simple threshold', function(done) {
    var test1 = new Alert({
      mode  : 'threshold',
      value : 32,
      func  : function(){ done(); }
    });
    //Done when current_value > 32
    while (current_value < 40) {
      test1.tick(current_value);
      current_value++;
    }
  });
  it('should detect reverse threshold', function(done) {  
    var current_value = 20;
    var test2 = new Alert({
      mode  : 'threshold',
      value : 32,
      func  : function(){ done(); },
      cmp   : function(a,b){ return (a < b); }
    });
    //Simulate the exception already launched
    test2.reached = 1;
    //Increment till it goes over 32 to reset the exception
    while (current_value < 40) {
      test2.tick(current_value);
      current_value++;
    }
    //Done when val < 32
    while (current_value > 20) {
      test2.tick(current_value);
      current_value--;
    }
  });
  it('should detect exception when avg > 90', function(done) {
    var current_value = 70;
    var i = 0;
    var test6 = new Alert({
      mode  : 'threshold-avg',
      value : 90,
      func  : function(){ done(); }
    });
    test6.start = true;
    //300 ticks at stable value == 5 mins of 1 sec ticks)
    while (i < 300) {
      test6.tick(current_value);
      i++;
    }
    current_value = 99;
    //ticks until detected (should be at 966)
    while(i < 967) {
      test6.tick(current_value);
      i++;
    }
  });
  it('should not detect exception over constant 2% max variation', function(done) {
    var current_value = 100;
    var test3 = new Alert({
      mode  : 'smart',
      func  : function() { done(new Error('Should not be called')); }
    });
    //Force instant start of data verification
    test3.start = true;
    
    var interval = setInterval(function() {
      test3.tick(current_value);
      //2% max deviation each step
      current_value += (Math.random() - 0.5) * (0.02 * current_value);
    }, 10);
    
    setTimeout(function() {
      clearInterval(interval);
      done();
    }, 1000);
  });
  it('should not detect (2 * value) spike every 30 values', function(done){
    var current_value = 100;
    var i = 0;
    var test4 = new Alert({
      mode  : 'smart',
      func  : function() { done(new Error('Should not be called')); }
    });
    //Force instant start of data verification
    test4.start = true;
    
    var interval = setInterval(function() {
      if (i % 30 == 29)
        test4.tick(2 * current_value)
      else
        test4.tick(current_value);
      //2% max deviation each step
      current_value += (Math.random() - 0.5) * (0.02 * current_value);
      i++;
    }, 10);
    
    setTimeout(function() {
      clearInterval(interval);
      done();
    }, 1000);
  });
  it('should detect 20 chained errors of 2 * value', function(done) {
  var current_value = 100;
    var test5 = new Alert({
      mode  : 'smart',
      func  : function() {
        //clear all, error was detected
        clearInterval(interval);
        clearTimeout(timeout);
        done();
      }
    });
    //Force instant start of data verification
    test5.start = true;
    
    var interval = setInterval(function() {
      test5.tick(current_value);
    }, 10);
    
    //Error Plateau Timeout
    var error = setTimeout(function() {
      for(var i = 0; i < 20; i++)
        test5.tick(current_value * 2);
    }, 600)    

    var timeout = setTimeout(function() {
      //Failed to detect the error plateau
      clearInterval(interval);
      done(new Error('Smart checker did not detect'));
    }, 1000);
  });
});
