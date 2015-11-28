var builders = require('recast').types.builders;

module.exports = function newFunctionExpression() {
  return builders.functionExpression(
    null,
    [],
    builders.blockStatement([])
  );
};
