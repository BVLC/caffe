


var pmx = require('../index.js');

pmx.initModule({
  alert_enabled : false
});

var Probe = pmx.probe();

// if null metric probe does not work
var slow_val = 0;

setInterval(function() {
  slow_val++;
  dt.set(slow_val);
}, 500);

var dt = Probe.metric({
  name : 'test',
  alert : {
    mode     : 'threshold',
    value    : 30,
    msg      : 'val too high'
  }
});
