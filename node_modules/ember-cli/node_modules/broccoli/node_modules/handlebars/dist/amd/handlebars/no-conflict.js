define(['exports', 'module'], function (exports, module) {
  /*global window */
  'use strict';

  module.exports = function (Handlebars) {
    /* istanbul ignore next */
    var root = typeof global !== 'undefined' ? global : window,
        $Handlebars = root.Handlebars;
    /* istanbul ignore next */
    Handlebars.noConflict = function () {
      if (root.Handlebars === Handlebars) {
        root.Handlebars = $Handlebars;
      }
    };
  };
});