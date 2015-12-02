var Notify    = require('../notify.js');
var Sample    = require('./sample.js');
var debug     = require('debug')('axm:alert:checker');

/**
 * Required:
 * opts.mode
 * opts.value
 *
 * Optional:
 * extra.name
 * opts.cmp
 * opts.interval
 * opts.msg
 * opts.sample
 */
var Alert = function(opts, extra) {
  var self = this;
  var cmp;

  if (typeof(opts.mode) === 'undefined')
    return console.error('[Probe][Metric] Mode undefined!');
  if (typeof(opts.cmp) === 'string') {
    switch(opts.cmp) {
      case '>':
      cmp = function(a,b) {
          return (a > b);
        };
        break;
      case '<':
       cmp = function(a, b) {
          return (a < b);
        };
        break;
      case '=':
         cmp = function(a,b) {
          return (a === b);
        };
        break;
      default:
        return console.error('[Probe][Metric] Mode does not exist!');
    }
    self.cmp_mode = opts.cmp;
    opts.cmp = null;
  }
  else {
    cmp = function(a,b) {
      return (a > b);
    };
    self.cmp_mode = '>';
  }
  switch(opts.mode) {
    case 'threshold':
      if (typeof(opts.value) === 'undefined')
        return console.error('[Probe][Metric][threshold] Val undefined!');
      this.cmp = opts.cmp || function(a,b) {
        return cmp(parseFloat(a), b);
      };
      break;
    case 'threshold-avg':
      if (typeof(opts.value) === 'undefined')
        return console.error('[Probe][Metric][threshold-avg] Val undefined!');
      this.sample = new Sample(opts.interval || 180);
      this.cmp = function(value, threshold) {
        this.sample.update(parseFloat(value));
        if (this.start) {
          if (typeof(opts.cmp) !== 'undefined' && opts.cmp !== null)
            return opts.cmp(this.sample.getMean(), threshold);
          return cmp(this.sample.getMean(), threshold);
        }
      };
      break;
    case 'smart':
      this.sample = new Sample(opts.interval || 300);
      this.small = new Sample(opts.sample || 30);
      this.cmp = function(value, threshold) {
        this.sample.update(parseFloat(value));
        this.small.update(parseFloat(value));
        //debug('Check', value, this.sample.getMean(), this.small.getMean());

        if (this.start)
          return (((this.small.getMean() - this.sample.getMean()) / this.sample.getMean()) > 0.2);
        return false;
      };
      break;
    default:
      return console.error('[Probe][Metric] Mode does not exist!');
  }
  this.mode = opts.mode;
  this.start = false;
  //Start the data checking after 30s (default)
  setTimeout(function() {
    self.start = true;
  }, opts.timeout || 30000);
  this.reached = 0;
  this.value =  opts.value || null;
  this.msg = opts.msg || ((extra && extra.name) ? ('Probe ' + extra.name + ' has reached value') : 'Alert value reached');
  this.func = opts.func || opts.action || null;
};

Alert.prototype.tick = function(value) {
  var self = this;
  if (this.reached === 0) {
    if (this.cmp(value, this.value)) {
      Notify.notify(this.msg + ' ' + this.value);
      // Delay a bit call to custom function to allow PM2 to receive notify msg
      if (this.func) setTimeout(function() {self.func(value)}, 50);
      this.reached = 1;
    }
  }
  else if (typeof(this.reached) !== 'undefined') {
    if (!this.cmp(value, this.value))
      this.reached = 0;
  }
};

Alert.prototype.serialize = function() {
  var self = this;

  var ret = {
    mode  : self.mode
  };

  if (this.mode == 'threshold') {
    ret.value = self.value;
    ret.cmp   = self.cmp_mode;
  }

  if (this.mode == 'threshold-avg') {
    ret.value    = self.value;
    ret.interval = self.sample._size;
    ret.cmp      = self.cmp_mode;
  }

  return ret;
};

module.exports = Alert;
