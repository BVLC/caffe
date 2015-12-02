var common = require('../fixtures/common'),
    assert = require('assert');

suite('AbstractParser class', function() {
  suite('given `123` string against multiple char matches', function() {
    var p = common.ap('123');

    test('should match `1` `2` `3` sequence', function() {
      assert.ok(p._match('1') && p._match('2') && p._match('3'));
    });

    test('and then fail on `1`', function() {
      assert.ok(!(p._match('1')));
    });
  });

  suite('given `123` against `124` or `123` sequence matches', function() {
    var p = common.ap('123');

    test('should not fail', function() {
      assert.ok(p._atomic(function() {
        return this._match('1') && this._match('2') && this._match('4');
      }) || p._atomic(function() {
        return this._match('1') && this._match('2') && this._match('3');
      }));
    });

    test('should choose `123` as intermediate value', function() {
      assert.equal(p._getIntermediate(), '123');
    });
  });

  suite('given `123` against `1` lookahead and `123` sequence', function() {
    var p = common.ap('123');

    test('should match', function() {
      assert.ok(p._atomic(function() {
        return this._match('1');
      }, true) || p._atomic(function() {
        return this._match('1') && this._match('2') && this._match('3');
      }));
    });
  });

  suite('given a nested list against nested list', function() {
    test('should match', function() {
      assert.ok(common.ap([
        '1',
        '2',
        ['3', '4'],
        '5'
      ])._atomic(function() {
        return this._match('1') && this._match('2') && this._list(function() {
          return this._match('3') && this._match('4');
        }) && this._match('5')
      }));
    });
  });

  suite('given `3` against simulates and groups matches', function() {
    test('should match', function() {
      assert.ok(common.ap('3')._atomic(function() {
        return this._atomic(function() {
          return this._simulate(['2'], function() {
            return this._simulate(['1'], function() {
              return this._match('1') && this._match('2') && this._match('3');
            });
          });
        })
      }));
    });
  });
});
