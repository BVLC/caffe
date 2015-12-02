
var pmx = require('..');


var conf = pmx.initModule({

  widget : {
    type             : 'generic',
    logo             : 'https://app.keymetrics.io/img/logo/keymetrics-300.png',

    // 0 = main element
    // 1 = secondary
    // 2 = main border
    // 3 = secondary border
    theme            : ['#141A1F', '#222222', '#3ff', '#3ff'],

    el : {
      probes  : true,
      actions : true
    },

    block : {
      actions : true,
      issues  : true,
      meta    : true
    }

    // Status
    // Green / Yellow / Red
  }
});


pmx.scopedAction('log streaming', function(data, emitter) {
  var i = setInterval(function() {
    emitter.send('this-is-a-line');
  }, 100);

  setTimeout(function() {

    emitter.end({success:true});
    clearInterval(i);
  }, 3000);
});

var spawn = require('child_process').spawn;

pmx.scopedAction('long running lsof', function(data, res) {
  var child = spawn('lsof', []);

  child.stdout.on('data', function(chunk) {
    chunk.toString().split('\n').forEach(function(line) {
      res.send(line);
    });
  });

  child.stdout.on('end', function(chunk) {
    res.end('end');
  });

});

pmx.scopedAction('with opts', function(data, res) {
  res.send(data);
  res.end('done');
});

pmx.scopedAction('throw err', function(data, res) {
  throw new Error('eroor!');
});

pmx.scopedAction('res.error', function(data, res) {
  res.error('this is a res.error');
});

pmx.action('simple action', function(reply) {
  return reply({success:true});
});

pmx.action('simple with arg', function(opts,reply) {
  return reply(opts);
});


var Probe = pmx.probe();

// if null metric probe does not work
var slow_val = 0;

setInterval(function() {
  slow_val++;
}, 500);

var dt = Probe.metric({
  name : 'test',
  value : function() {
    return slow_val;
  },
  alert : {
    mode     : 'threshold',
    val      : 30,
    //interval : 60, // seconds
    msg      : 'val too hight'
  }
});
