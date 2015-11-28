'use strict';

var commandProperties = [
  'name',
  'description',
  'aliases',
  'works',
  'availableOptions',
  'anonymousOptions'
];
var blueprintProperties = [
  'name',
  'description',
  'availableOptions',
  'anonymousOptions',
  'overridden'
];

function forEachWithProperty(properties, forEach, context) {
  return properties.filter(function(key) {
    return this[key] !== undefined;
  }, context).forEach(forEach, context);
}

module.exports = {
  command: {
    forEachWithProperty: function(forEach, context) {
      return forEachWithProperty(commandProperties, forEach, context);
    }
  },
  blueprint: {
    forEachWithProperty: function(forEach, context) {
      return forEachWithProperty(blueprintProperties, forEach, context);
    }
  }
};
