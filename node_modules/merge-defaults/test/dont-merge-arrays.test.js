var assert = require('assert');
var _ = require('lodash');
_.mergeDefaults = require('../');


describe('mergeDefaults', function() {

  describe('dont merge arrays', function() {

    var X, Y, result;

    beforeEach(function() {
      X = {
        z: 1,
        a: 2,
        b: 3,
        d: {},
        e: []
      };
      Y = {
        a: 1,
        b: 22,
        c: 33,
        d: {
          x: 10
        },
        e: ['a','b']
      };
      result = _.mergeDefaults(X, Y);
    });

    it('should return an object', function() {
      assert(typeof result === 'object');
    });
    it('should NOT MERGE ARRAYS in sub-objects', function() {
      assert.deepEqual(result.e, []);
    });
  });

  describe('complex (recursive)', function() {

    var X, Y, result;

    beforeEach(function() {
      X = {
        views: {
          foo: {},
          blueprints: {
            someArray: ['z'],
            enabled: true
          }
        },
        connections: {},
        z: 1,
        a: 2,
        b: 3,
        d: {}
      };
      Y = {
        views: {
          locales: ['en', 'es'],

        },
        controllers: {
          foo: {
            bar: 'asdf'
          },
          blueprints: {
            someArray: ['a','b'],
            enabled: false
          }
        },
        connections: {
          mysql: {
            host: 'localhost',
            port: 1835913851
          }
        },
        a: 1,
        b: 22,
        c: 33,
        d: {
          x: 10
        }
      };
      result = _.mergeDefaults(X, Y);
    });

    it('should return an object', function() {
      assert(typeof result === 'object');
    });
    it('should NOT MERGE ARRAYS in sub-objects', function() {
      assert.deepEqual(result.views.blueprints.someArray, ['z']);
    });
  });
});