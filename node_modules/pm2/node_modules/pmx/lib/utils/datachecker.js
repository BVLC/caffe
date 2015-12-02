
var probe     = require('../Probe.js');
var Histogram = require('./probes/Histogram.js');
var debug     = require('debug')('axm:smart:checker');

 /**
  * new smart({
  *   refresh   : function,   Returns monitored value
  *   callback  : function,   Called when error is detected
  *   dev       : 0.2,        Deviation that triggers the error
  *   timer     : 100<ms>,    Value refresh interval
  *   launch    : 10000<ms>,  Timelength after which monitoring begins
  *   ceil      : 5           Number of times an error is detected before triggering callback
  *   calcDEv   : mean        How to compute the deviation
  * });
  */

function dataChecker(opts) {
  var self = this;
  this._counter = 0;

  opts = opts || {};

  if (typeof(opts.refresh) !== 'function')
    throw new Error('Refresh not defined or not a function');

  this._refresh = opts.refresh;

  this._monitored = false;
  this._launch = opts.launch || 10000;
  this._ceil = opts.ceil || 5;
  this._timer = opts.timer || 1000;
  this._dev   = opts.dev || 0.2;
  this._callback = opts.callback || null;
  this._histogram = new Histogram();
  this._calcDev = opts.calcDev || 'ema';

  /*
   * Select function to compute deviation
   */
  var func = {
    ema    : this.normalDev,
    mean   : this.medianDev
  }
  this.computeDev = func[this._calcDev];

  /**
   * Display some probe if we need to debug
   */
  if (opts.debug === true) {
    if (opts.probes.indexOf('val') != -1) {
      this._metric2 = new probe.metric({
        name  : 'Value',
        value : function() {
          return self._refresh();
        }
      });
    }
    if (opts.probes.indexOf('ema') != -1) {
      this._metric3 = new probe.metric({
        name  : 'EMA',
        value : function() {
          return self._histogram.getEma();
        }
      });
    }
  }
  /**
    * Delay before monitoring starts
    */
  setTimeout(function() {
    self._monitored = true;
  }, this._launch);
  this.start();
};

//Calculate deviation of current value compared to EMA
dataChecker.prototype.normalDev = function() {
  return ((this._refresh() - this._histogram.getEma()) / this._histogram.getEma() > this._dev);
}

//Calculate deviation of current EMA compared to Mean
dataChecker.prototype.medianDev = function() {
  return ((this._histogram.getEma() - this._histogram._calculateMean()) / this._histogram._calculateMean() > this._dev);
}

dataChecker.prototype.stop = function() {
  clearInterval(this._interval_timer);
};

dataChecker.prototype.start = function() {
  var self = this;

  debug('Checker started');

  this._interval_timer = setInterval(function() {
    self._histogram.update(self._refresh());

    debug('Check', self._refresh(), self._histogram._calculateMean(), self._histogram.getEma());

    if (self._monitored === true && self.computeDev() === true) {
      self._counter++;
      /**
        * Triggers callback after N consecutive errors, then resets the counter
        */
      if (self._counter >= self._ceil) {
        debug('Anomaly detected', self._histogram.getEma(), self._refresh());
        self._callback();
        self.counter = 0;
      }
    }
    else
      self._counter = 0;
  }, self._timer);
};

module.exports = dataChecker;
