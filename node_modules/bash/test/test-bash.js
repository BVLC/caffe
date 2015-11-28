var common = require('./common');
var assert = common.assert;
var bash = common.require('bash');

(function testEscape() {
  assert.equal(bash.escape('hello _-world2'), 'hello\\ \\_-world2');
  assert.equal(bash.escape('a'), 'a');
  assert.equal(bash.escape('a', 'b'), 'a b');
  assert.equal(bash.escape('a b'), 'a\\ b');
  assert.equal(bash.escape(5), '5');
  assert.equal(bash.escape(''), "''");
  assert.equal(bash.escape(undefined), "''");
  assert.equal(bash.escape(null), "''");
  assert.equal(bash.escape(0), '0');
  assert.equal(bash.escape('Geisendörfer'), 'Geisend\\örfer');
})();

(function testArgs() {
  (function testOneOption() {
    var r = bash.args({foo: 'bar'}, '--', '=');
    assert.equal(r, '--foo=bar');
  })();

  (function testTwoOptions() {
    var r = bash.args({a: 1, b: 2}, '--', '=');
    assert.equal(r, '--a=1 --b=2');
  })();

  (function testValueEscaping() {
    var r = bash.args({a: 'hey you'}, '--', '=');
    assert.equal(r, '--a=hey\\ you');
  })();

  (function testNullValueActsAsFlag() {
    var r = bash.args({a: null}, '--', '=');
    assert.equal(r, '--a');
  })();

  (function testTrueValueActsAsFlag() {
    var r = bash.args({a: true}, '--', '=');
    assert.equal(r, '--a');
  })();

  (function testArrayValueCreatesMultipleArgs() {
    var r = bash.args({a: [1, 2]}, '--', '=');
    assert.equal(r, '--a=1 --a=2');
  })();

  (function testArrayMap() {
    var r = bash.args([{a: 1}, {a: 2, b: 3}], '--', '=');
    assert.equal(r, '--a=1 --a=2 --b=3');
  })();

  (function testAlternatePrefixSuffix() {
    var r = bash.args({a: 1}, '-', ' ');
    assert.equal(r, '-a 1');
  })();
})();
