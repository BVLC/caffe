
var EventEmitter = require('events').EventEmitter;
var fileset      = require('../');
var assert       = require('assert');
var test         = require('./helper');

// Given a **.md pattern
test('Sync API - Given a **.md pattern', function() {
  return {
    'should return the list of matching file in this repo': function(em) {
      var results = fileset.sync('*.md', 'node_modules/**/*.md');

      assert.ok(Array.isArray(results), 'should be an array');
      assert.ok(results.length, 'should return at least one element');
      assert.equal(results.length, 2, 'actually, should return only two');

      em.emit('end');
    }
  }
});

test('Sync API - Given a *.md and **.js pattern, and two exclude', function() {
  return {
    'should return the list of matching file in this repo': function(em) {
      var results = fileset.sync('*.md *.js', 'CHANGELOG.md node_modules/**/*.md node_modules/**/*.js');

      assert.ok(Array.isArray(results), 'should be an array');
      assert.ok(results.length, 'should return at least one element');

      assert.equal(results.length, 6, 'actually, should return only six');

      em.emit('end');
    }
  }
});

test.run();
