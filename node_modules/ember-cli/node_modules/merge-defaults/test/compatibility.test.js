var _ = require('lodash');
var should = require('should');
_.defaultsDeep = require('../');



/**
 * Purpose:
 * This test exists to make sure I didn't break anything when
 * using mergeDefaults to override `_.defaults`.
 */

// From Lodash core tests:
// https://github.com/lodash/lodash/blob/master/test/test.js#L1843
describe('Test that _.mergeDefaults is backwards compatible with _.defaults \n', function() {
  it('should assign properties of a source object if missing on the destination object', function() {
    deepEqual(_.defaultsDeep({
      'a': 1
    }, {
      'a': 2,
      'b': 2
    }), {
      'a': 1,
      'b': 2
    });
  });

  it('should assign own source properties', function() {
    function Foo() {
      this.a = 1;
      this.c = 3;
    }

    Foo.prototype.b = 2;
    deepEqual(_.defaultsDeep({
      'c': 2
    }, new Foo()), {
      'a': 1,
      'c': 2
    });
  });

  it('should accept multiple source objects', function() {
    var expected = {
      'a': 1,
      'b': 2,
      'c': 3
    };
    deepEqual(_.defaultsDeep({
      'a': 1,
      'b': 2
    }, {
      'b': 3
    }, {
      'c': 3
    }), expected);
    deepEqual(_.defaultsDeep({
      'a': 1,
      'b': 2
    }, {
      'b': 3,
      'c': 3
    }, {
      'c': 2
    }), expected);
  });

  it('should not overwrite `null` values', function() {
    var actual = _.defaultsDeep({
      'a': null
    }, {
      'a': 1
    });
    strictEqual(actual.a, null);
  });

  it('should overwrite `undefined` values', function() {
    var actual = _.defaultsDeep({
      'a': undefined
    }, {
      'a': 1
    });
    strictEqual(actual.a, 1);
  });

  it('should not error on `null` or `undefined` sources (test in IE < 9)', function() {
    try {
      deepEqual(_.defaultsDeep({
        'a': 1
      }, null, undefined, {
        'a': 2,
        'b': 2
      }), {
        'a': 1,
        'b': 2
      });
    } catch (e) {
      throw e;
    }
  });
});


// helper methods
function strictEqual(x, y) {
  return should(x).equal(y);
}

function deepEqual(x, y) {
  return should(x).eql(y);
}