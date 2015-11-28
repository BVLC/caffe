'use strict';

var SilentError = require('silent-error');

module.exports = function normalizeEntityName(entityName) {

  if (!entityName) {
    throw new SilentError('The `ember generate <entity-name>` command requires an ' +
                          'entity name to be specified. ' +
                          'For more details, use `ember help`.');
  }

  var trailingSlash = /(\/$|\\$)/;
  if(trailingSlash.test(entityName)) {
    throw new SilentError('You specified "' + entityName + '", but you can\'t use a ' +
                          'trailing slash as an entity name with generators. Please ' +
                          're-run the command with "' + entityName.replace(trailingSlash, '') + '".');
  }

  return entityName;
};
