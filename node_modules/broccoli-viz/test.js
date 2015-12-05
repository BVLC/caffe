var flatten = require('./flatten');
var dot = require('./dot')
var assert = require('assert');
var rank = require('./rank');
var processGraph  = require('./process');

function toJSON() { return this; }

function node(n) {
  n.toJSON = toJSON;
  return n;
}

describe('flatten', function() {
  it('works with one root', function(){
    var node = {
      subtrees: []
    };

    var result = flatten(node);

    assert.equal(result.length, 1);
    assert.deepEqual(result, [node]);
  });

  it('works with 2 levels', function() {
    var a = node({ id: 'a', subtrees: [] });
    var b = node({ id: 'b', subtrees: [] });
    var c = node({ id: 'c', subtrees: [a, b] });

    var result = flatten(c);

    assert.equal(result.length, 3);
    assert.deepEqual(result, [c, a, b]);
  });

  it('works with messy tree and 4 levels', function() {
    var a = node({ id: 'a', subtrees: [] });
    var b = node({ id: 'b', subtrees: [] });
    var c = node({ id: 'c', subtrees: [a, b] });
    var d = node({ id: 'd', subtrees: [c] });
    var e = node({ id: 'e', subtrees: [d, b] });

    var result = flatten(e);

    assert.equal(result.length, 5);
    assert.deepEqual(result, [e , d, c, a, b]);
  });
});

describe('dot', function() {
  it('works with one root', function(){
    var a = node({
      id: 'a',
      subtrees: [],
      selfTime: 1000000,
      totalTime: 1000000
    });

    var result = dot(processGraph(a));

    assert.equal(result, 'digraph G { ratio = \"auto\" a [shape=circle, style=\"dotted\", label=\" a self time (1ms)\n total time (1ms)\" ]\n}');
  });

  it('works with self and total time', function(){
    var a = node({
      id: 'a',
      subtrees: [],
      selfTime: 1000000,
      totalTime: 1000000
    });

    var result = dot(processGraph(a));

    assert.equal(result, 'digraph G { ratio = \"auto\" a [shape=circle, style=\"dotted\", label=\" a self time (1ms)\n total time (1ms)\" ]\n}');
  });
});


describe('rank', function() {

  it('empty', function(){
    var g = node({
      id: 1,
      subtrees: [],
      selfTime: 0,
      totalTime: 5
    });

    var path = flatten(rank(g)).map(mapById);

    assert.deepEqual(path, [1]);
  });

  function mapById(node) {
    return node.id;
  }


  function byRank(a, b) {
    return a.rank - b.rank;
  }

  it('one path', function(){
    var f = node({
      id: 1,
      subtrees: [],
      selfTime: 0,
      totalTime: 5
    });

    var g = node({
      id: 2,
      subtrees: [f],
      selfTime: 0,
      totalTime: 5
    });

    var path = flatten(rank(g));

    assert.deepEqual(path.map(mapById), [2, 1]);
  });

  it('slight more complex graph', function() {
    /*
            ┌───────────────┐
            │#1 TotalTime: 5│
            └───────────────┘
                    │
        ╔═══════════╩──────────┐
        ║                      │
        ▼                      ▼
┌───────────────┐      ┌───────────────┐
│#2 TotalTime: 4│      │#3 TotalTime: 3│
└───────────────┘      └───────────────┘
        ║                      │
        ║                      │
        ▼                      ▼
┌───────────────┐      ┌───────────────┐
│#4 TotalTime: 2│      │#5 TotalTime: 2│
└───────────────┘      └───────────────┘
        ║                      │
        ║                      │
        ▼                      ▼
┌───────────────┐     ┌────────────────┐
│#6 TotalTime: 2│     │#7 TotalTime: 1 │
└───────────────┘     └────────────────┘
    */

    var a = node({
      id: 7,
      subtrees: [],
      selfTime: 0,
      totalTime: 1
    });

    var b = node({
      id: 6,
      subtrees: [],
      selfTime: 0,
      totalTime: 2
    });

    var c = node({
      id: 5,
      subtrees: [a],
      selfTime: 0,
      totalTime: 2
    });

    var d = node({
      id: 4,
      subtrees: [b],
      selfTime: 0,
      totalTime: 2
    });

    var e = node({
      id: 3,
      subtrees: [c],
      selfTime: 0,
      totalTime: 3
    });

    var f = node({
      id: 2,
      subtrees: [d],
      selfTime: 0,
      totalTime: 4
    });

    var g = node({
      id: 1,
      subtrees: [e, f],
      selfTime: 0,
      totalTime: 5
    });

    var path = flatten(rank(g)).sort(byRank);

    assert.deepEqual(path.map(mapById), [1,2,4,6,3,5,7]);
  });

  it('slight more complex graph \w tie', function() {
    /*
            ┌───────────────┐
            │#1 TotalTime: 5│
            └───────────────┘
                    │
        ╔═══════════╩──────────┐
        ║                      │
        ▼                      ▼
┌───────────────┐      ┌───────────────┐
│#2 TotalTime: 4│      │#3 TotalTime: 4│
└───────────────┘      └───────────────┘
        ║                      │
        ║                      │
        ▼                      ▼
┌───────────────┐      ┌───────────────┐
│#4 TotalTime: 2│      │#5 TotalTime: 2│
└───────────────┘      └───────────────┘
        ║                      │
        ║                      │
        ▼                      ▼
┌───────────────┐     ┌────────────────┐
│#6 TotalTime: 2│     │#7 TotalTime: 2 │
└───────────────┘     └────────────────┘
    */

    var a = node({
      id: 7,
      subtrees: [],
      selfTime: 0,
      totalTime: 2
    });

    var b = node({
      id: 6,
      subtrees: [],
      selfTime: 0,
      totalTime: 2
    });

    var c = node({
      id: 5,
      subtrees: [a],
      selfTime: 0,
      totalTime: 2
    });

    var d = node({
      id: 4,
      subtrees: [b],
      selfTime: 0,
      totalTime: 2
    });

    var e = node({
      id: 3,
      subtrees: [c],
      selfTime: 0,
      totalTime: 4
    });

    var f = node({
      id: 2,
      subtrees: [d],
      selfTime: 0,
      totalTime: 4
    });

    var g = node({
      id: 1,
      subtrees: [e, f],
      selfTime: 0,
      totalTime: 5
    });

    var path = flatten(rank(g)).sort(byRank);

    assert.deepEqual(path.map(mapById), [1,3,5,7,2,4,6]);
  });

});

