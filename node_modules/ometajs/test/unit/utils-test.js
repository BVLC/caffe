var common = require('../fixtures/common'),
    assert = require('assert');

suite('utils', function() {
  suite('expressionify routine', function() {
    function unit(source, expected) {
      assert.equal(common.expressionify(source), expected);
    }

    test('should add `return` in correct places', function() {
      unit('a', 'a');
      unit('a;b', 'function(){a;return b}.call(this)');
      unit('{ x = 1 }', 'function(){return x=1}.call(this)');
      unit('{ x = 1; y = 2 }', 'function(){x=1;return y=2}.call(this)');
      unit('switch (x) { case 1: return x; }', 'switch(x){case 1:return x}');
    });
  });

  suite('localify routine', function() {
    function unit(source, expected) {
      var result = common.localify(source, 0);
      assert.deepEqual(result.vars, expected.vars);
      assert.equal(result.before, expected.before);
      assert.equal(result.afterSuccess, expected.afterSuccess);
      assert.equal(result.afterFail, expected.afterFail);
    }

    test('should create correct statements', function() {
      unit('x = 1, y = 1, this[z] = 1', {
        vars: ['$l0', '$l1', '$l2', '$l3', '$l4'],
        before: [
          '$l0=x,x=1,$l1=y,y=1,' +
          '$l2=this,$l3=z,$l4=$l2[$l3],$l2[$l3]=1,true'
        ],
        afterSuccess: ['x=$l0,y=$l1,$l2[$l3]=$l4,true'],
        afterFail: ['x=$l0,y=$l1,$l2[$l3]=$l4,false']
      });
    });
  });
});
