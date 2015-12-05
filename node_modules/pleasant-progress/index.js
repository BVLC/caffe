'use strict';

module.exports = PleasantProgress;
function PleasantProgress(options) {
  // Options
  options = options || {};
  this.rate = options.rate || 500;
  this.stream = options.stream || process.stdout;

  // Defaults
  this.isRunning = false;
  this.progress = 0;
}

PleasantProgress.prototype.forPromise = function(message, promise) {
  this.start(message);

  try {
    return promise.then(function(value) {
      this.stop();
      return value;
    }, function(reason) {
      this.stop();
      throw reason;
    });
  } finally {
    this.stop();
  }
};

PleasantProgress.prototype.start = function(message, stepString) {
  this.message = message || '';
  this.stepString = stepString || '.';
  this.stop();
  this.isRunning = true;
  this.print();
  this.stepInterval = setInterval(this.step.bind(this), this.rate);
};

PleasantProgress.prototype.stop = function(printWithFullStepString) {
  if (this.isRunning) {
    this.isRunning = false;
    clearInterval(this.stepInterval);
    this.clear();
    if (printWithFullStepString) {
      this.progress = 3;
      this.print();
      this.stream.write('\n');
    }
  }
};

PleasantProgress.prototype.step = function() {
  this.progress = (this.progress >= 3) ? 0 : (this.progress + 1);
  this.clear();
  this.print();
};

PleasantProgress.prototype.print = function() {
  var msg = this.message + repeat(this.stepString, this.progress);
  this.stream.write(msg);
};

PleasantProgress.prototype.clear = function() {
  if (process.stdout.isTTY) {
    this.stream.clearLine();
    this.stream.cursorTo(0);
  }
};

function repeat(string, times) {
  return (times === 0) ? '' : string + repeat(string, times - 1);
}
