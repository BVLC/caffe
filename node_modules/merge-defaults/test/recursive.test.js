var assert = require('assert');
var should = require('should');
var _ = require('lodash');
_.mergeDefaults = require('../');

describe('mergeDefaults', function() {

  describe('basic (recursive)', function() {

    var X, Y, result;

    beforeEach(function() {
      X = {
        z: 1,
        a: 2,
        b: 3,
        d: {}
      };
      Y = {
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
      should(result).be.an.Object;
    });
    it('should have the expected values from first arg', function() {
      should(result).have.property('z', 1);
      should(result).have.property('a', 2);
      should(result).have.property('b', 3);
    });
    it('should have the expected values from second arg', function() {
      should(result).have.property('c', 33);
    });
    it('should have recursively merged the sub-objects', function() {
      should(result).have
        .property('d').with.a
        .property('x', 10);
    });
  });

  describe('complex (recursive)', function() {

    var X, Y, result;

    beforeEach(function() {
      X = {
        views: {
          foo: {},
          blueprints: {
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
      should(result).be.an.Object;
    });
    it('should have the expected values from first arg', function() {
      should(result).have.property('z', 1);
      should(result).have.property('a', 2);
      should(result).have.property('b', 3);
    });
    it('should have the expected values from second arg', function() {
      should(result).have.property('c', 33);
    });
    it('should have recursively merged the sub-objects', function() {
      should(result).have
        .property('d').with.a
        .property('x', 10);
    });

    it('should still work even when shit gets crazy', function() {
      // .views.locales
      should(result).have
        .property('views').with.a
        .property('locales');

      // .views.blueprints.enabled = true
      should(result).have
        .property('views').with
        .property('blueprints').with
        .property('enabled', true);

      // controllers.foo.bar
      should(result).have
        .property('controllers').with
        .property('foo').with
        .property('bar');

      // connections.mysql.host = 'localhost'
      should(result).have
        .property('connections').with
        .property('mysql').with
        .property('host', 'localhost');
    });

  });

  describe('3-level-deep merge', function() {

    var X = {
      foo: {
        a:1,
        b:2,
        bar: {
          a:1,
          b:2
        }
      }
    };
    var Y = {
      foo: {
        a:100,
        c:3,
        bar: {
          a:10,
          c:3
        }
      }
    };
    var result = _.mergeDefaults(X, Y);

    it('should retain the values in X (first arg)', function (){
      assert.equal(result.foo.a, 1);
      assert.equal(result.foo.bar.a, 1);
      assert.equal(result.foo.bar.b, 2);
    });

    it('should receive new values from Y (second arg)', function () {
      assert.equal(result.foo.bar.c, 3);
    });
  });


  describe('4-level-deep merge', function() {

    var X = {
      baz: {
        foo: {
          a:1,
          b:2,
          bar: {
            a:1,
            b:2
          }
        }
      }
    };
    var Y = {
      baz: {
        foo: {
          a:100,
          c:3,
          bar: {
            a:10,
            c:3
          }
        }
      }
    };
    var result = _.mergeDefaults(X, Y);

    it('should retain the values in X (first arg)', function (){
      assert.equal(result.baz.foo.a, 1);
      assert.equal(result.baz.foo.bar.a, 1);
      assert.equal(result.baz.foo.bar.b, 2);
    });

    it('should receive new values from Y (second arg)', function () {
      assert.equal(result.baz.foo.bar.c, 3);
    });
  });
});