var builders         = require('recast').types.builders;
var routeOptionsNode = require('./route-options-node');


module.exports = function routeNode(name, options) {
  options = options || {};

  var node = builders.expressionStatement(
    builders.callExpression(
      builders.memberExpression(
        builders.thisExpression(),
        builders.identifier('route'),
        false
      ),
      [builders.literal(name)]
    )
  );

  var optionsNode = routeOptionsNode(options);
  if (optionsNode) {
    node.expression.arguments.push(optionsNode);
  }

  return node;
};
