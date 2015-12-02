# tape

tap-producing test harness for node and browsers

[![browser support](https://ci.testling.com/substack/tape.png)](http://ci.testling.com/substack/tape)

[![build status](https://secure.travis-ci.org/substack/tape.png)](http://travis-ci.org/substack/tape)

![tape](http://substack.net/images/tape_drive.png)

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

# pretty reporters

The default TAP output is good for machines and humans that are robots.

If you want a more colorful / pretty output there are lots of modules on npm
that will output something pretty if you pipe TAP into them:

 - https://github.com/scottcorgan/tap-spec
 - https://github.com/scottcorgan/tap-dot
 - https://github.com/substack/faucet
 - https://github.com/juliangruber/tap-bail
 - https://github.com/kirbysayshi/tap-browser-color
 - https://github.com/gummesson/tap-json
 - https://github.com/gummesson/tap-min
 - https://github.com/calvinmetcalf/tap-nyan
 - https://www.npmjs.org/package/tap-pessimist
 - https://github.com/toolness/tap-prettify
 - https://github.com/shuhei/colortape
 - https://github.com/aghassemi/tap-xunit

To use them, try `node test/index.js | tap-spec` or pipe it into one
of the modules of your choice!

# uncaught exceptions

By default, uncaught exceptions in your tests will not be intercepted, and will cause tape to crash. If you find this behavior undesirable, use [tape-catch](https://github.com/michaelrhodes/tape-catch) to report any exceptions as TAP errors.

# methods

The assertion methods in tape are heavily influenced or copied from the methods
in [node-tap](https://github.com/isaacs/node-tap).

```
var test = require('tape')
```

## test([name], [opts], cb)

Create a new test with an optional `name` string and optional `opts` object. 
`cb(t)` fires with the new test object `t` once all preceeding tests have
finished. Tests execute serially.

Available `opts` options are:
- opts.skip = true/false. See test.skip.
- opts.timeout = 500. Set a timeout for the test, after which it will fail. 
  See test.timeoutAfter.

If you forget to `t.plan()` out how many assertions you are going to run and you
don't call `t.end()` explicitly, your test will hang.

## test.skip(name, cb)

Generate a new test that will be skipped over.

## t.plan(n)

Declare that `n` assertions should be run. `t.end()` will be called
automatically after the `n`th assertion. If there are any more assertions after
the `n`th, or after `t.end()` is called, they will generate errors.

## t.end(err)

Declare the end of a test explicitly. If `err` is passed in `t.end` will assert
that it is falsey.

## t.fail(msg)

Generate a failing assertion with a message `msg`.

## t.pass(msg)

Generate a passing assertion with a message `msg`.

## t.timeoutAfter(ms)

Automatically timeout the test after X ms.

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

## t.equal(actual, expected, msg)

Assert that `actual === expected` with an optional description `msg`.

Aliases: `t.equals()`, `t.isEqual()`, `t.is()`, `t.strictEqual()`,
`t.strictEquals()`

## t.notEqual(actual, expected, msg)

Assert that `actual !== expected` with an optional description `msg`.

Aliases: `t.notEquals()`, `t.notStrictEqual()`, `t.notStrictEquals()`,
`t.isNotEqual()`, `t.isNot()`, `t.not()`, `t.doesNotEqual()`, `t.isInequal()`

## t.deepEqual(actual, expected, msg)

Assert that `actual` and `bexpected` have the same structure and nested values using
[node's deepEqual() algorithm](https://github.com/substack/node-deep-equal)
with strict comparisons (`===`) on leaf nodes and an optional description
`msg`.

Aliases: `t.deepEquals()`, `t.isEquivalent()`, `t.same()`

## t.notDeepEqual(actual, expected, msg)

Assert that `actual` and `expected` do not have the same structure and nested values using
[node's deepEqual() algorithm](https://github.com/substack/node-deep-equal)
with strict comparisons (`===`) on leaf nodes and an optional description
`msg`.

Aliases: `t.notEquivalent()`, `t.notDeeply()`, `t.notSame()`,
`t.isNotDeepEqual()`, `t.isNotDeeply()`, `t.isNotEquivalent()`,
`t.isInequivalent()`

## t.deepLooseEqual(actual, expected, msg)

Assert that `actual` and `expected` have the same structure and nested values using
[node's deepEqual() algorithm](https://github.com/substack/node-deep-equal)
with loose comparisons (`==`) on leaf nodes and an optional description `msg`.

Aliases: `t.looseEqual()`, `t.looseEquals()`

## t.notDeepLooseEqual(actual, expected, msg)

Assert that `actual` and `expected` do not have the same structure and nested values using
[node's deepEqual() algorithm](https://github.com/substack/node-deep-equal)
with loose comparisons (`==`) on leaf nodes and an optional description `msg`.

Aliases: `t.notLooseEqual()`, `t.notLooseEquals()`

## t.throws(fn, expected, msg)

Assert that the function call `fn()` throws an exception. `expected`, if present, must be a `RegExp` or `Function`.

## t.doesNotThrow(fn, expected, msg)

Assert that the function call `fn()` does not throw an exception.

## t.test(name, cb)

Create a subtest with a new test handle `st` from `cb(st)` inside the current
test `t`. `cb(st)` will only fire when `t` finishes. Additional tests queued up
after `t` will not be run until all subtests finish.

## var htest = test.createHarness()

Create a new test harness instance, which is a function like `test()`, but with
a new pending stack and test state.

By default the TAP output goes to `console.log()`. You can pipe the output to
someplace else if you `htest.createStream().pipe()` to a destination stream on
the first tick.

## test.only(name, cb)

Like `test(name, cb)` except if you use `.only` this is the only test case
that will run for the entire process, all other test cases using tape will
be ignored

## var stream = test.createStream(opts)

Create a stream of output, bypassing the default output stream that writes
messages to `console.log()`. By default `stream` will be a text stream of TAP
output, but you can get an object stream instead by setting `opts.objectMode` to
`true`.

### tap stream reporter

You can create your own custom test reporter using this `createStream()` api:

``` js
var test = require('tape');
var path = require('path');

test.createStream().pipe(process.stdout);

process.argv.slice(2).forEach(function (file) {
    require(path.resolve(file));
});
```

You could substitute `process.stdout` for whatever other output stream you want,
like a network connection or a file.

Pass in test files to run as arguments:

```
$ node tap.js test/x.js test/y.js
TAP version 13
# (anonymous)
not ok 1 should be equal
  ---
    operator: equal
    expected: "boop"
    actual:   "beep"
  ...
# (anonymous)
ok 2 should be equal
ok 3 (unnamed assert)
# wheee
ok 4 (unnamed assert)

1..4
# tests 4
# pass  3
# fail  1
```

### object stream reporter

Here's how you can render an object stream instead of TAP:

``` js
var test = require('tape');
var path = require('path');

test.createStream({ objectMode: true }).on('data', function (row) {
    console.log(JSON.stringify(row))
});

process.argv.slice(2).forEach(function (file) {
    require(path.resolve(file));
});
```

The output for this runner is:

```
$ node object.js test/x.js test/y.js
{"type":"test","name":"(anonymous)","id":0}
{"id":0,"ok":false,"name":"should be equal","operator":"equal","actual":"beep","expected":"boop","error":{},"test":0,"type":"assert"}
{"type":"end","test":0}
{"type":"test","name":"(anonymous)","id":1}
{"id":0,"ok":true,"name":"should be equal","operator":"equal","actual":2,"expected":2,"test":1,"type":"assert"}
{"id":1,"ok":true,"name":"(unnamed assert)","operator":"ok","actual":true,"expected":true,"test":1,"type":"assert"}
{"type":"end","test":1}
{"type":"test","name":"wheee","id":2}
{"id":0,"ok":true,"name":"(unnamed assert)","operator":"ok","actual":true,"expected":true,"test":2,"type":"assert"}
{"type":"end","test":2}
```

# install

With [npm](https://npmjs.org) do:

```
npm install tape
```

# license

MIT
