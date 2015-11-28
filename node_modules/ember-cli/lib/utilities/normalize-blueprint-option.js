'use strict';

var path = require('path');

module.exports = function normalizeBlueprintOption(blueprint) {
  return blueprint[0] === '.' ? path.resolve(process.cwd(), blueprint) : blueprint;
};
