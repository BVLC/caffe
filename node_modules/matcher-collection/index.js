'use strict';

var Minimatch = require('minimatch').Minimatch;

module.exports = MatcherCollection;

function MatcherCollection(matchers) {
  this.matchers = matchers.map(function(matcher) {
    return typeof matcher === 'string' ? new Minimatch(matcher) : matcher;
  });
}

MatcherCollection.prototype.match = function(value) {
  for (var i = 0; i < this.matchers.length; i++) {
    if (this.matchers[i].match(value)) {
      return true;
    }
  }

  return false;
};

MatcherCollection.prototype.mayContain = function(value) {
  var parts = value.split(/\/|\\/g).filter(Boolean);

  for (var i = 0; i < this.matchers.length; i++) {
    var matcher = this.matchers[i];
    for (var j = 0; j < matcher.set.length; j++) {
      if (matcher.matchOne(parts, matcher.set[j], true)) {
        return true;
      }
    }
  }

  return false;
};
