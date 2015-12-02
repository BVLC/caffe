var os = require('os')

var stats = require('./lib/stats')

var wrapper = function(stat_type) {

  return function(pid, options, cb) {

    if(typeof options == 'function') {
      cb = options
      options = {}
    }

    return stats[stat_type](pid, options, cb)
  }
}

var pusage = {
  darwin: wrapper('ps'),
  sunos: wrapper('ps'),
  freebsd: wrapper('ps'),
  win: wrapper('win'),
  linux: wrapper('proc'),
  aix: wrapper('ps'),
  unsupported: function(pid, options, cb) {
    cb = typeof options == 'function' ? options : cb

    cb(new Error(os.platform()+' is not supported yet, please fire an issue (https://github.com/soyuka/pidusage)'))
  }
}

var platform = os.platform();
platform = platform.match(/^win/) ? 'win' : platform; //nor is windows a winner...
platform = pusage[platform] ? platform : 'unsupported';

exports.stat = function() {
  pusage[platform].apply(stats, [].slice.call(arguments));
};

exports.unmonitor = function(pid) {
  delete stats.history[pid];
};

exports._history = stats.history;
