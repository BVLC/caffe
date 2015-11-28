'use strict';

module.exports = MockAnalytics;
function MockAnalytics() {
  this.tracks = [];
  this.trackTimings = [];
  this.trackErrors = [];
}

MockAnalytics.prototype = Object.create({});
MockAnalytics.prototype.track = function(arg) {
  this.tracks.push(arg);
};

MockAnalytics.prototype.trackTiming = function(arg) {
  this.trackTimings.push(arg);
};

MockAnalytics.prototype.trackError = function(arg) {
  this.trackErrors.push(arg);
};

MockAnalytics.prototype.constructor = MockAnalytics;
