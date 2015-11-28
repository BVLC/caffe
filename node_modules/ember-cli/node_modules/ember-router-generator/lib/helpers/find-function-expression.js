module.exports = function findFunctionExpression(nodes) {
  var node;

  for (var i = 0; i < nodes.length; i++) {
    node = nodes[i];

    if (node.type === 'FunctionExpression') {
      return node;
    }
  }

  return false;
};
