'use strict';

function visit(vertex, fn, visited, path) {
  var name = vertex.name,
    vertices = vertex.incoming,
    names = vertex.incomingNames,
    len = names.length,
    i;
  if (!visited) {
    visited = {};
  }
  if (!path) {
    path = [];
  }
  if (visited.hasOwnProperty(name)) {
    return;
  }
  path.push(name);
  visited[name] = true;
  for (i = 0; i < len; i++) {
    visit(vertices[names[i]], fn, visited, path);
  }
  fn(vertex, path);
  path.pop();
}

function DAG() {
  this.names = [];
  this.vertices = {};
}

DAG.prototype.add = function(name) {
  if (!name) { return; }
  if (this.vertices.hasOwnProperty(name)) {
    return this.vertices[name];
  }
  var vertex = {
    name: name,
    incoming: {},
    incomingNames: [],
    hasOutgoing: false,
    value: null
  };

  this.vertices[name] = vertex;
  this.names.push(name);
  return vertex;
};

DAG.prototype.map = function(name, value) {
  this.add(name).value = value;
};

DAG.prototype.addEdge = function(fromName, toName) {
  if (!fromName || !toName || fromName === toName) {
    return;
  }
  var from = this.add(fromName), to = this.add(toName);
  if (to.incoming.hasOwnProperty(fromName)) {
    return;
  }
  function checkCycle(vertex, path) {
    if (vertex.name === toName) {
      throw new Error('cycle detected: ' + toName + ' <- ' + path.join(' <- '));
    }
  }
  visit(from, checkCycle);
  from.hasOutgoing = true;
  to.incoming[fromName] = from;
  to.incomingNames.push(fromName);
};

DAG.prototype.topsort = function(fn) {
  var visited = {},
    vertices = this.vertices,
    names = this.names,
    len = names.length,
    i, vertex;
  for (i = 0; i < len; i++) {
    vertex = vertices[names[i]];
    if (!vertex.hasOutgoing) {
      visit(vertex, fn, visited);
    }
  }
};

DAG.prototype.addEdges = function(name, value, before, after) {
  var i;
  this.map(name, value);
  if (before) {
    if (typeof before === 'string') {
      this.addEdge(name, before);
    } else {
      for (i = 0; i < before.length; i++) {
        this.addEdge(name, before[i]);
      }
    }
  }
  if (after) {
    if (typeof after === 'string') {
      this.addEdge(after, name);
    } else {
      for (i = 0; i < after.length; i++) {
        this.addEdge(after[i], name);
      }
    }
  }
};

module.exports = DAG;
