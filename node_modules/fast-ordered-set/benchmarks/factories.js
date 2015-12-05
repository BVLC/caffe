'use strict';

var FastOrderdSet = require('../');
// var OrderedSet = require('ordered-set'); // doesn't work
var SetLib = require('set');
var AbstractSet = require('abstract-set');
var SetJs = require('set-js');
var equalityFunction = function(a,b){return a == b;};
var lesserFunction = function(a,b){return a < b;};
var SpecializedSet = require('specialized-set')(equalityFunction, lesserFunction);
var SetCollection = require('set-collection');
var assert = require('assert');
var factories = [];
var initial5 =  Array.apply(null, {length: 5}).map(Number.call, Number);
var initial50 = Array.apply(null, {length: 50}).map(Number.call, Number);

var fastOrderedSetEmpty = new FastOrderdSet();
var fastOrderedSet5 = new FastOrderdSet(initial5);
var fastOrderedSet50 = new FastOrderdSet(initial50);

factories.push({
  name: 'fast-ordered-set',
  create: function fastOrderedSet() {
    return new FastOrderdSet();
  },
  create5: function fastOrderedSet() {
    return new FastOrderdSet(initial5);
  },
  create50: function fastOrderedSet() {
    return new FastOrderdSet(initial50);
  },

  hasEmpty: function() {
    return fastOrderedSetEmpty.has('missing');
  },
  hasHit5: function() {
    return fastOrderedSet5.has(3);
  },
  hasHit50: function() {
    return fastOrderedSet50.has(25);
  },
  hasMiss50: function() {
    return fastOrderedSet50.has(51);
  }
});

var es2015OrderedSetEmpty = new Set();
var es2015OrderedSet5 = new Set(initial5);
var es2015OrderedSet50 = new Set(initial50);

factories.push({
  name: 'es2015',
  create: function es2015() {
    return new Set();
  },
  create5: function es2015() {
    return new Set(initial5);
  },
  create50: function es2015() {
    return new Set(initial50);
  },
  hasEmpty: function() {
    return es2015OrderedSetEmpty.has('missing');
  },
  hasHit5: function() {
    return es2015OrderedSet5.has(3);
  },
  hasHit50: function() {
    return es2015OrderedSet50.has(25);
  },
  hasMiss50: function() {
    return es2015OrderedSet50.has(51);
  }
});

var setLibEmpty = new SetLib();
var setLibHit5 = new SetLib(initial5);
var setLibHit50 = new SetLib(initial50);

factories.push({
  name: 'set',
  create: function setLib() {
    return new SetLib();
  },
  create5: function setLib() {
    return new SetLib(initial5);
  },
  create50: function setLib() {
    return new SetLib(initial50);
  },
  hasEmpty: function() {
    return setLibEmpty.has('missing');
  },
  hasHit5: function() {
    return setLibHit5.has(3);
  },
  hasHit50: function() {
    return setLibHit50.has(25);
  },
  hasEmpty: function() {
    return setLibEmpty.has('missing');
  },
  hasMiss50: function() {
    return setLibHit50.has(51);
  }
});

var setJsEmpty = new SetJs();
var setJsHit5 = new SetJs(initial5);
var setJsHit50 = new SetJs(initial50);

factories.push({
  name: 'set-js',
  create: function setJs() {
    return new SetJs();
  },
  create5: function setJs() {
    return new SetJs(initial5);
  },
  create50: function setJs() {
    return new SetJs(initial50);
  },
  hasEmpty: function() {
    return setJsEmpty.has('missing');
  },
  hasHit5: function() {
    return setJsHit5.has(3);
  },
  hasHit50: function() {
    return setJsHit5.has(25);
  },
  hasMiss50: function() {
    return setJsHit5.has(51);
  }
});

var specializedSetEmpty = new SpecializedSet();
var specializedSetHit5 = new SpecializedSet(initial5);
var specializedSetHit50 = new SpecializedSet(initial50);

factories.push({
  name: 'specialized-set',
  create: function specializedSet() {
    return new SpecializedSet();
  },
  create5: function specializedSet() {
    return new SpecializedSet(initial5);
  },
  create50: function specializedSet() {
    return new SpecializedSet(initial50);
  },
  hasEmpty: function() {
    return specializedSetEmpty.has('missing');
  },
  hasHit5: function() {
    return specializedSetHit5.has(3);
  },
  hasHit50: function() {
    return specializedSetHit50.has(25);
  },
  hasMiss50: function() {
    return specializedSetHit50.has(50);
  }
});

var setCollectionEmpty = new SetCollection();
var setCollectionHit5  = new SetCollection(initial5);
var setCollectionHit50  = new SetCollection(initial50);

factories.push({
  name: 'set-collection',
  create: function setCollection() {
    return new SetCollection();
  },
  create5: function setCollection() {
    return new SetCollection(initial5);
  },
  create50: function setCollection() {
    return new SetCollection(initial50);
  },
  hasEmpty: function() {
    return setCollectionEmpty.has('missing');
  },
  hasHit5: function() {
    return setCollectionHit5.has(3);
  },
  hasHit50: function() {
    return setCollectionHit50.has(25);
  },
  hasMiss50: function() {
    return setCollectionHit50.has(51);
  }
});

module.exports = {
  all: factories,
  byName: byName,
  byTest: byTest
};

function byName(name) {
  return factories.filter(function(f) {
    return f.name === name;
  })[0];
}

function byTest(name) {
  return factories.map(function(f) {
    var copy = { };

    for (var prop in f) {
      copy[prop] = f[prop];
    }

    copy.fn = f[name];

    return copy;
  });
}

assert.equal(byName('fast-ordered-set').create().size, 0);
assert.equal(byName('es2015').create().size, 0);
// is currently broken...
//assert(orderedSet() instanceof OrderedSet);
assert.equal(byName('set').create().size(), 0);
assert.equal(byName('set-js').create().size(), 0);
assert.equal(byName('specialized-set').create().size, 0);
assert.equal(byName('set-collection').create().count, 0);

assert.equal(byName('fast-ordered-set').create5().size, 5);
assert.equal(byName('es2015').create5().size, 5);
// is currently broken...
//assert(orderedSet() instanceof OrderedSet);
assert.equal(byName('set').create5().size(), 5);
assert.equal(byName('set-js').create5().size(), 5);
assert.equal(byName('specialized-set').create5().size, 5);
assert.equal(byName('set-collection').create5().count, 5);

