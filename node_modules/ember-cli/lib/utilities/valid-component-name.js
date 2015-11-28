'use strict';
var SilentError = require('silent-error');

module.exports = function(entityName) {
  if(! /\-/.test(entityName)) {
    throw new SilentError('You specified "' + entityName + '", but in order to prevent ' +
                          'clashes with current or future HTML element names, you must include ' +
                          'a hyphen in the component name.');
  }

  return entityName;
};
