'use strict';

// largest to smaller
function byTotalTime(x, y) {
  return y.totalTime - x.totalTime;
}

function RankedNode(node, level) {
  this.id = node.id;
  this.level = level;
  this.node = node;
  this.subtrees = [];
  this.stats = {}; // Bucket for additional stats and metrics
}

RankedNode.prototype.toJSON  = function() {
  var json = this.node.toJSON();
  json.level = this.level;
  json.stats = this.stats;
  return json;
};

module.exports = function level(root, theLevel) {
  var currentLevel = arguments.length === 1 ? 0 : theLevel;

  // TODO: add ranking system
  var leveled = new RankedNode(root, currentLevel);

  var subtrees = root.subtrees;
  if (subtrees.length === 0 ) { return leveled; }

  leveled.subtrees = subtrees.sort(byTotalTime).map(function(unleveled, i) {
    return level(unleveled, currentLevel + i);
  });

  return leveled;
};
