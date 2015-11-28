'use strict';
var Set = require('./set');

module.exports = function flatten() {
  var result, root;

  if (arguments.length === 1) {
    result = new Set();
    root = arguments[0];
  } else {
    result = arguments[0];
    root = arguments[1];
  }

  result.add(root);

  for (var i = 0; i < root.subtrees.length; i++) {
    flatten(result, root.subtrees[i]);
  }

  return result.values;
};
