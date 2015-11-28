'use strict';

var Filter = require('broccoli-filter');
var CleanCSS = require('clean-css');

function CleanCSSFilter(inputTree, options) {
  if (!(this instanceof CleanCSSFilter)) {
    return new CleanCSSFilter(inputTree, options);
  }

  this.inputTree = inputTree;
  this.options = options || {};
}

CleanCSSFilter.prototype = Object.create(Filter.prototype);
CleanCSSFilter.prototype.constructor = CleanCSSFilter;

CleanCSSFilter.prototype.extensions = ['css'];
CleanCSSFilter.prototype.targetExtension = 'css';

CleanCSSFilter.prototype.processString = function(str) {
  return new CleanCSS(this.options).minify(str);
};

module.exports = CleanCSSFilter;
