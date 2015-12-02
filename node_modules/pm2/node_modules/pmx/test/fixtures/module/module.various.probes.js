

var pmx = require('../../..');

var conf = pmx.initModule({
  alert_enabled : true
}, function() {
  var probe = pmx.probe();
  var slow_val = 0;

  var dt = probe.metric({
    name  : 'probe-test',
    alert : {
      mode     : 'threshold-avg',
      value    : 15,
      interval : 5,
      msg      : 'val too high'
    }
  });


  var TIME_INTERVAL = 1000;

  var oldTime = process.hrtime();

  var histogram = probe.histogram({
    name        : 'delay',
    measurement : 'mean',
    unit        : 'ms'
  });

  setInterval(function() {
    var newTime = process.hrtime();
    var delay = (newTime[0] - oldTime[0]) * 1e3 + (newTime[1] - oldTime[1]) / 1e6 - TIME_INTERVAL;
    oldTime = newTime;
    histogram.update(delay);
  }, TIME_INTERVAL);



  /**
   * Meter
   */
  /**
   * Probe system #3 - Meter
   *
   * Probe things that are measured as events / interval.
   */
  var meter = probe.meter({
    name    : 'req/min',
    seconds : 60
  });


  setInterval(function() {
    meter.mark();
  }, 100);


  /**
   * Probe system #4 - Counter
   *
   * Measure things that increment or decrement
   */
  var counter = probe.counter({
    name : 'Downloads'
  });

  setInterval(function() {
    counter.inc();
  }, 10);

});
