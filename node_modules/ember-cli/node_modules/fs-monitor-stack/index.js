'use strict';
var fs = require('fs');
module.exports = FSMonitor;

function FSMonitor() {
  this.stack = [];
  this.state = 'idle';
  this.stats = [];
  this.blacklist = ['createReadStream', 'createWriteStream', 'ReadStream', 'WriteStream'];
}


FSMonitor.prototype._start = function() {
  this.state = 'active';
  this._attach();
};

FSMonitor.prototype._stop = function() {
  this.state = 'idle';
  this._detach();
};

FSMonitor.prototype.shouldMeasure = function() {
  return this.state === 'active';
};

FSMonitor.prototype.push = function(node) {
  this.stack.push(node);

  if (this.length === 1) {
    this._start();
  }
};

FSMonitor.prototype.statsFor = function(node) {
  var id = node.id;

  if (this.stats.length <= id) {
    return null;
  } else {
    return this.stats[id];
  }
};

FSMonitor.prototype.totalStats = function() {
  var result = Object.create(null);

  this.stats.forEach(function(stat) {
    Object.keys(stat).forEach(function(key) {
      var m = result[key] = (result[key] || new Metric());
      m.count += (stat[key].count);
      m.time += (stat[key].time);
    });
  });

  return result;
};

function Metric() {
  this.count = 0;
  this.time = 0;
  this.startTime = undefined;
}

Metric.prototype.start = function() {
  this.startTime = process.hrtime();
  this.count++;
};

Metric.prototype.stop = function() {
  var now = process.hrtime();

  this.time += (now[0] - this.startTime[0]) * 1e9 + (now[1] - this.startTime[1]);
  this.startTime = undefined;
};

Metric.prototype.toJSON = function() {
  return {
    count: this.count,
    time: Math.round(this.time / 1e4) / 1e2
  };
};

FSMonitor.prototype._measure = function(name, original, context, args) {
  if (this.state !== 'active') {
    throw new Error('Cannot measure if the monitor is not active');
  }
  var id = this.top.id;

  if (typeof id !== 'number') {
    throw new Error('EWUT: encountered unexpected node without an id....');
  }
  var metrics = this.stats[id] = this.stats[id] || Object.create(null);
  var m = metrics[name] = metrics[name] || new Metric();

  m.start();

  // TODO: handle async
  try {
    return original.apply(context, args);
  } finally {
    m.stop();
  }
};

FSMonitor.prototype._attach = function() {
  var monitor = this;

  for (var member in fs) {
    if (this.blacklist.indexOf(member) === -1) {
      var old = fs[member];
      if (typeof old === 'function') {
        fs[member] = (function(old, member) {
          return function() {
            if (monitor.shouldMeasure()) {
              return monitor._measure(member, old, fs, arguments);
            } else {
              return old.apply(fs, arguments);
            }
          };
        }(old, member));

        fs[member].__restore = function() {
          fs[member] = old;
        };
      }
    }
  }
};

FSMonitor.prototype._detach = function() {
  for (var member in fs) {
    if (typeof old === 'function') {
      fs[member].__restore();
    }
  }
};

FSMonitor.prototype.reset = function() {
  this.stats.length = 0;
};

FSMonitor.prototype.pop = function(node) {
  this.stack.pop();

  if (this.length === 0) {
    this._stop();
  }
};

Object.defineProperty(FSMonitor.prototype, 'length', {
  get: function() {
    return this.stack.length;
  }
});

Object.defineProperty(FSMonitor.prototype, 'top', {
  get: function() {
    return this.stack[this.stack.length - 1];
  }
});
