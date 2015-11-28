'use strict';

var upstreamMergeTrees = require('broccoli-merge-trees');

module.exports = function(inputTree, options) {
  options = options || {};
  options.description = options.annotation;
  var tree = upstreamMergeTrees(inputTree, options);

  tree.description = options && options.description;

  return tree;
};
