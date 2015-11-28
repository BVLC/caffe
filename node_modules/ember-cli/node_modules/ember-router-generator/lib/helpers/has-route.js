var recast = require('recast');

function isRoute(node) {
  var callee = node.expression.callee;
  return node.expression.type === 'CallExpression' && callee.type === 'MemberExpression' && callee.property.name === 'route';
}

module.exports = function hasRoute(name, routes) {
  var nodePath = false;

  recast.visit(routes, {
    visitExpressionStatement: function(path) {
      var node = path.node;

      if (isRoute(node) &&
          node.expression.arguments[0].value === name) {
        nodePath = path;

        return false;
      } else {
        this.traverse(path);
      }
    }
  });

  return nodePath;
};
