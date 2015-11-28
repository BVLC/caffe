'use strict';
/*jshint node:true*/
var normalizeEntityName = require('ember-cli-normalize-entity-name');

module.exports = {
  description: 'Generates a helper function.',
  normalizeEntityName: function(entityName) {
    return normalizeEntityName(entityName);
  }
};
