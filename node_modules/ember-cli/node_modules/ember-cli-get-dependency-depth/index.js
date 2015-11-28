'use strict';

module.exports = function getDependencyDepth(name) {
  var nameParts = name.split('/');
  var depth = '../..';

  return nameParts.reduce(function(prev) {
    return prev + '/..';
  }, depth);
};
