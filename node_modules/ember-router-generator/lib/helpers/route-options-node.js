var builders = require('recast').types.builders;

module.exports = function routeOptionNode(options) {
  options = options || {};

  var node = builders.objectExpression([]);
  var properties = [];

  if (options.path) {
    properties.push(
      builders.property(
        'init',
        builders.identifier('path'),
        builders.literal(options.path)
      )
    );
  }

  if (options.hasOwnProperty('resetNamespace')) {
    properties.push(
      builders.property(
        'init',
        builders.identifier('resetNamespace'),
        builders.literal(options.resetNamespace)
      )
    );
  }

  if (!properties.length) {
    return null;
  }

  node.properties = node.properties.concat(properties);

  return node;
};
