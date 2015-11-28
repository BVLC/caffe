'use strict';

var semver = require('semver');

var LOWER_RANGE = '0.12.0';
var UPPER_RANGE = '6.0.0';

function PlatformChecker(version) {
  this.version = version;
  this.isValid = this.checkIsValid();
  this.isUntested = this.checkIsUntested();
  this.isDeprecated = this.checkIsDeprecated();
}

PlatformChecker.prototype.checkIsValid = function() {
  return semver.satisfies(this.version, '>=' + LOWER_RANGE + ' <' + UPPER_RANGE);
};

PlatformChecker.prototype.checkIsDeprecated = function() {
  return semver.satisfies(this.version, '<' + LOWER_RANGE);
};

PlatformChecker.prototype.checkIsUntested = function() {
  return semver.satisfies(this.version, '>=' + UPPER_RANGE);
};

module.exports = PlatformChecker;

