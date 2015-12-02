# tape

tap-producing test harness for node and browsers

[![browser support](https://ci.testling.com/substack/tape.png)](http://ci.testling.com/substack/tape)

[![build status](https://secure.travis-ci.org/substack/tape.png)](http://travis-ci.org/substack/tape)

![tape](http://substack.net/images/tape_drive.png)

# browser compatibility

chrome, firefox, opera, safari, IE6, IE7, IE8, IE9

using browserify@1.16.5

# example

``` js
var test = require('tape');

test('timing test', function (t) {
    t.plan(2);
    
    t.equal(typeof Date.now, 'function');
    var start = Date.now();
    
    setTimeout(function () {
        t.equal(Date.now() - start, 100);
    }, 100);
});
```

```
$ node example/timing.js
TAP version 13
# timing test
ok 1 should be equal
not ok 2 should be equal
  ---
    operator: equal
    expected: 100
    actual:   107
  ...

1..2
# tests 2
# pass  1
# fail  1
```

# methods

The assertion methods in tape are heavily influenced or copied from the methods
in [node-tap](https://github.com/isaacs/node-tap).

```
var test = require('tape')
```

## test(name, cb)

Create a new test with an optional `name` string. `cb(t)` fires with the new
test object `t` once all preceeding tests have finished. Tests execute serially.

If you forget to `t.plan()` out how many assertions you are going to run and you
don't call `t.end()` explicitly, your test will hang.

## t.plan(n)

Declare that `n` assertions should be run. `t.end()` will be called
automatically after the `n`th assertion. If there are any more assertions after
the `n`th, or after `t.end()` is called, they will generate errors.

## t.end()

Declare the end of a test explicitly.

## t.fail(msg)

Generate a failing assertion with a message `msg`.

## t.pass(msg)

Generate a passing assertion with a message `msg`.

## t.skip(msg)
 
Generate an assertion that will be skipped over.

## t.ok(value, msg)

Assert that `value` is truthy with an optional description message `msg`.

Aliases: `t.true()`, `t.assert()`

## t.notOk(value, msg)

Assert that `value` is falsy with an optional description message `msg`.

Aliases: `t.false()`, `t.notok()`

## t.error(err, msg)

Assert that `err` is falsy. If `err` is non-falsy, use its `err.message` as the
description message.

Aliases: `t.ifError()`, `t.ifErr()`, `t.iferror()`

## t.equal(a, b, msg)

Assert that `a === b` with an optional description `msg`.

Aliases: `t.equals()`, `t.isEqual()`, `t.is()`, `t.strictEqual()`,
`t.strictEquals()`

## t.notEqual(a, b, msg)

Assert that `a !== b` with an optional description `msg`.

Aliases: `t.notEquals()`, `t.notStrictEqual()`, `t.notStrictEquals()`,
`t.isNotEqual()`, `t.isNot()`, `t.not()`, `t.doesNotEqual()`, `t.isInequal()`

## t.deepEqual(a, b, msg)

Assert that `a` and `b` have the same structure and nested values using
[node's deepEqual() algorithm](https://github.com/substack/node-deep-equal)
with strict comparisons (`===`) on leaf nodes and an optional description
`msg`.

Aliases: `t.deepEquals()`, `t.isEquivalent()`, `t.same()`

## t.notDeepEqual(a, b, msg)

Assert that `a` and `b` do not have the same structure and nested values using
[node's deepEqual() algorithm](https://github.com/substack/node-deep-equal)
with strict comparisons (`===`) on leaf nodes and an optional description
`msg`.

Aliases: `t.notEquivalent()`, `t.notDeeply()`, `t.notSame()`,
`t.isNotDeepEqual()`, `t.isNotDeeply()`, `t.isNotEquivalent()`,
`t.isInequivalent()`

## t.deepLooseEqual(a, b, msg)

Assert that `a` and `b` have the same structure and nested values using
[node's deepEqual() algorithm](https://github.com/substack/node-deep-equal)
with loose comparisons (`==`) on leaf nodes and an optional description `msg`.

Aliases: `t.looseEqual()`, `t.looseEquals()`

## t.notDeepLooseEqual(a, b, msg)

Assert that `a` and `b` do not have the same structure and nested values using
[node's deepEqual() algorithm](https://github.com/substack/node-deep-equal)
with loose comparisons (`==`) on leaf nodes and an optional description `msg`.

Aliases: `t.notLooseEqual()`, `t.notLooseEquals()`

## t.throws(fn, expected, msg)

Assert that the function call `fn()` throws an exception.

## t.doesNotThrow(fn, expected, msg)

Assert that the function call `fn()` does not throw an exception.

## t.test(name, cb)

Create a subtest with a new test handle `st` from `cb(st)` inside the current
test `t`. `cb(st)` will only fire when `t` finishes. Additional tests queued up
after `t` will not be run until all subtests finish.

## var htest = test.createHarness()

Create a new test harness instance, which is a function like `test()`, but with
a new pending stack and test state.

By default the TAP output goes to `process.stdout` or `console.log()` if the
environment doesn't have `process.stdout`. You can pipe the output to someplace
else if you `test.stream.pipe()` to a destination stream on the first tick.

## test.only(name, cb)

Like `test(name, cb)` except if you use `.only` this is the only test case
that will run for the entire process, all other test cases using tape will
be ignored

# install

With [npm](https://npmjs.org) do:

```
npm install tape
```

# license

MIT
