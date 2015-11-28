// usage: DUMP_BROCCOLI_TREES=true broccoli serve
// results: ./broccoli-tree.json
// converting to dot: node $BROCCOLI_PATH/scripts/graph broccoli-trees.json > graph.dot
// visualizing: dot -Tpng graph.dot -o graph.png

function formatTime(time) {
  return Math.floor(time / 1e6) + 'ms';
}

// replace this with actual math
function selfTimeColor(n) {
  var number = n/ 1e6;

  if (number > 5000) { return 1; }
  if (number > 2000) { return 2; }
  if (number > 1000) { return 3; }
  if (number >  250) { return 4; }
  if (number >  100) { return 5; }
  if (number >   50) { return 6; }
  if (number >   10) { return 7; }
  if (number >    5) { return 8; }
  return 9;
}

function edgeColor(r) {
  var level = r + 1;
  return Math.max(1, Math.min(level, 4));
}

function penWidth(level) {
  if (level === 0) return 3;
  if (level === 1) return 1.5;
  return 0.5;
}

function nodesById(nodes) {
  var result = new Array(nodes.length);
  nodes.forEach(function(node) {
    result[node.id] = node;
  });
  return result;
}
// reds8
module.exports = function dot(nodes) {
  var out = 'digraph G {';
  out += ' ratio = "auto"';

  var byId = nodesById(nodes);

  nodes.map(function(node) {
    return node.toJSON();
  }).forEach(function(node) {
    out += ' ' + node.id;
    var annotation = node.annotation || node.description;
    if (annotation) {
      annotation = annotation.replace('(', '\n(');

      var shape, style;

      if (annotation.indexOf('Merge') > -1) {
        shape = 'circle';
        style = 'dashed';
      } else if (annotation.indexOf('Funnel') > -1) {
        shape = 'box';
        style = 'dashed';
      } else {
        shape = 'box';
        style = 'solid';
      }

      out += ' [shape=' + shape + ', style=' + style + ', colorscheme="rdylbu9", color=' + selfTimeColor(node.selfTime) +', label=" ' +
         node.id + ' \n' +
         annotation  + '\n' +
        ' self time (' + formatTime(node.selfTime) + ') \n' +
        ' total time (' + formatTime(node.totalTime) + ')\n "]';

    } else {
      out += ' [shape=circle, style="dotted", label=" ' + node.id +
        ' self time (' + formatTime(node.selfTime) +
        ')\n total time (' + formatTime(node.totalTime) +
        ')" ]';
    }

    out += '\n';
    node.subtrees.forEach(function(child) {
      var level = node.level + byId[child].level;
      out += ' ' + child + ' -> ' + node.id + '[penwidth=' + penWidth(level) + ' ] \n';
    });
  });
  out += '}';
  return out;
};
