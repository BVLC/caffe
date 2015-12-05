var rank = require('./rank');
var flatten = require('./flatten');

module.exports = function(g) {
  return flatten(rank(g));
};
