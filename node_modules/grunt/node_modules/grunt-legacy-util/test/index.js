'use strict';

var util = require('../');

var fs = require('fs');
var path = require('path');

var Tempfile = require('temporary/lib/file');

exports['util.callbackify'] = {
  'return': function(test) {
    test.expect(1);
    // This function returns a value.
    function add(a, b) {
      return a + b;
    }
    util.callbackify(add)(1, 2, function(result) {
      test.equal(result, 3, 'should be the correct result.');
      test.done();
    });
  },
  'callback (sync)': function(test) {
    test.expect(1);
    // This function accepts a callback which it calls synchronously.
    function add(a, b, done) {
      done(a + b);
    }
    util.callbackify(add)(1, 2, function(result) {
      test.equal(result, 3, 'should be the correct result.');
      test.done();
    });
  },
  'callback (async)': function(test) {
    test.expect(1);
    // This function accepts a callback which it calls asynchronously.
    function add(a, b, done) {
      setTimeout(done.bind(null, a + b), 0);
    }
    util.callbackify(add)(1, 2, function(result) {
      test.equal(result, 3, 'should be the correct result.');
      test.done();
    });
  }
};

exports['util'] = {
  'error': function(test) {
    test.expect(9);
    var origError = new Error('Original error.');

    var err = util.error('Test message.');
    test.ok(err instanceof Error, 'Should be an Error.');
    test.equal(err.name, 'Error', 'Should be an Error.');
    test.equal(err.message, 'Test message.', 'Should have the correct message.');

    err = util.error('Test message.', origError);
    test.ok(err instanceof Error, 'Should be an Error.');
    test.equal(err.name, 'Error', 'Should be an Error.');
    test.equal(err.message, 'Test message.', 'Should have the correct message.');
    test.equal(err.origError, origError, 'Should reflect the original error.');

    var newError = new Error('Test message.');
    err = util.error(newError, origError);
    test.equal(err, newError, 'Should be the passed-in Error.');
    test.equal(err.origError, origError, 'Should reflect the original error.');
    test.done();
  },
  'linefeed': function(test) {
    test.expect(1);
    if (process.platform === 'win32') {
      test.equal(util.linefeed, '\r\n', 'linefeed should be operating-system appropriate.');
    } else {
      test.equal(util.linefeed, '\n', 'linefeed should be operating-system appropriate.');
    }
    test.done();
  },
  'normalizelf': function(test) {
    test.expect(1);
    if (process.platform === 'win32') {
      test.equal(util.normalizelf('foo\nbar\r\nbaz\r\n\r\nqux\n\nquux'), 'foo\r\nbar\r\nbaz\r\n\r\nqux\r\n\r\nquux', 'linefeeds should be normalized');
    } else {
      test.equal(util.normalizelf('foo\nbar\r\nbaz\r\n\r\nqux\n\nquux'), 'foo\nbar\nbaz\n\nqux\n\nquux', 'linefeeds should be normalized');
    }
    test.done();
  }
};

exports['util.spawn'] = {
  setUp: function(done) {
    this.script = path.resolve('test/fixtures/spawn.js');
    done();
  },
  'exit code 0': function(test) {
    test.expect(6);
    util.spawn({
      cmd: process.execPath,
      args: [ this.script, 0 ],
    }, function(err, result, code) {
      test.equals(err, null);
      test.equals(code, 0);
      test.equals(result.stdout, 'stdout');
      test.equals(result.stderr, 'stderr');
      test.equals(result.code, 0);
      test.equals(String(result), 'stdout');
      test.done();
    });
  },
  'exit code 0, fallback': function(test) {
    test.expect(6);
    util.spawn({
      cmd: process.execPath,
      args: [ this.script, 0 ],
      fallback: 'ignored if exit code is 0'
    }, function(err, result, code) {
      test.equals(err, null);
      test.equals(code, 0);
      test.equals(result.stdout, 'stdout');
      test.equals(result.stderr, 'stderr');
      test.equals(result.code, 0);
      test.equals(String(result), 'stdout');
      test.done();
    });
  },
  'non-zero exit code': function(test) {
    test.expect(7);
    util.spawn({
      cmd: process.execPath,
      args: [ this.script, 123 ],
    }, function(err, result, code) {
      test.ok(err instanceof Error);
      test.equals(err.message, 'stderr');
      test.equals(code, 123);
      test.equals(result.stdout, 'stdout');
      test.equals(result.stderr, 'stderr');
      test.equals(result.code, 123);
      test.equals(String(result), 'stderr');
      test.done();
    });
  },
  'non-zero exit code, fallback': function(test) {
    test.expect(6);
    util.spawn({
      cmd: process.execPath,
      args: [ this.script, 123 ],
      fallback: 'custom fallback'
    }, function(err, result, code) {
      test.equals(err, null);
      test.equals(code, 123);
      test.equals(result.stdout, 'stdout');
      test.equals(result.stderr, 'stderr');
      test.equals(result.code, 123);
      test.equals(String(result), 'custom fallback');
      test.done();
    });
  },
  'cmd not found': function(test) {
    test.expect(3);
    util.spawn({
      cmd: 'nodewtfmisspelled',
    }, function(err, result, code) {
      test.ok(err instanceof Error);
      test.equals(code, 127);
      test.equals(result.code, 127);
      test.done();
    });
  },
  'cmd not found, fallback': function(test) {
    test.expect(4);
    util.spawn({
      cmd: 'nodewtfmisspelled',
      fallback: 'use a fallback or good luck'
    }, function(err, result, code) {
      test.equals(err, null);
      test.equals(code, 127);
      test.equals(result.code, 127);
      test.equals(String(result), 'use a fallback or good luck');
      test.done();
    });
  },
  'cmd not in path': function(test) {
    test.expect(6);
    var win32 = process.platform === 'win32';
    util.spawn({
      cmd: 'test\\fixtures\\exec' + (win32 ? '.cmd' : '.sh'),
    }, function(err, result, code) {
      test.equals(err, null);
      test.equals(code, 0);
      test.equals(result.stdout, 'done');
      test.equals(result.stderr, '');
      test.equals(result.code, 0);
      test.equals(String(result), 'done');
      test.done();
    });
  },
  'cmd not in path (with cwd)': function(test) {
    test.expect(6);
    var win32 = process.platform === 'win32';
    util.spawn({
      cmd: './exec' + (win32 ? '.cmd' : '.sh'),
      opts: {cwd: 'test/fixtures'},
    }, function(err, result, code) {
      test.equals(err, null);
      test.equals(code, 0);
      test.equals(result.stdout, 'done');
      test.equals(result.stderr, '');
      test.equals(result.code, 0);
      test.equals(String(result), 'done');
      test.done();
    });
  },
  'grunt': function(test) {
    test.expect(3);
    util.spawn({
      grunt: true,
      args: [ '--gruntfile', 'test/fixtures/Gruntfile-print-text.js', 'print:foo' ],
    }, function(err, result, code) {
      test.equals(err, null);
      test.equals(code, 0);
      test.ok(/^OUTPUT: foo/m.test(result.stdout), 'stdout should contain output indicating the grunt task was run.');
      test.done();
    });
  },
  'grunt (with cwd)': function(test) {
    test.expect(3);
    util.spawn({
      grunt: true,
      args: [ '--gruntfile', 'Gruntfile-print-text.js', 'print:foo' ],
      opts: {cwd: 'test/fixtures'},
    }, function(err, result, code) {
      test.equals(err, null);
      test.equals(code, 0);
      test.ok(/^OUTPUT: foo/m.test(result.stdout), 'stdout should contain output indicating the grunt task was run.');
      test.done();
    });
  },
  'grunt passes execArgv': function(test) {
    test.expect(3);
    util.spawn({
      cmd: process.execPath,
      args: [ '--harmony', process.argv[1], '--gruntfile', 'test/fixtures/Gruntfile-execArgv.js'],
    }, function(err, result, code) {
      test.equals(err, null);
      test.equals(code, 0);
      test.ok(/^OUTPUT: --harmony/m.test(result.stdout), 'stdout should contain passed-through process.execArgv.');
      test.done();
    });
  },
  'grunt result.toString() with error': function(test) {
    // grunt.log.error uses standard out, to be fixed in 0.5.
    test.expect(4);
    util.spawn({
      grunt: true,
      args: [ 'nonexistentTask' ]
    }, function(err, result, code) {
      test.ok(err instanceof Error, 'Should be an Error.');
      test.equal(err.name, 'Error', 'Should be an Error.');
      test.equals(code, 3);
      test.ok(/Warning: Task "nonexistentTask" not found./m.test(result.toString()), 'stdout should contain output indicating the grunt task was (attempted to be) run.');
      test.done();
    });
  },
  'custom stdio stream(s)': function(test) {
    test.expect(6);
    var stdoutFile = new Tempfile();
    var stderrFile = new Tempfile();
    var stdout = fs.openSync(stdoutFile.path, 'a');
    var stderr = fs.openSync(stderrFile.path, 'a');
    var child = util.spawn({
      cmd: process.execPath,
      args: [ this.script, 0 ],
      opts: {stdio: [null, stdout, stderr]},
    }, function(err, result, code) {
      test.equals(code, 0);
      test.equals(String(fs.readFileSync(stdoutFile.path)), 'stdout\n', 'Child process stdout should have been captured via custom stream.');
      test.equals(String(fs.readFileSync(stderrFile.path)), 'stderr\n', 'Child process stderr should have been captured via custom stream.');
      stdoutFile.unlinkSync();
      stderrFile.unlinkSync();
      test.equals(result.stdout, '', 'Nothing will be passed to the stdout string when spawn stdio is a custom stream.');
      test.done();
    });
    test.ok(!child.stdout, 'child should not have a stdout property.');
    test.ok(!child.stderr, 'child should not have a stderr property.');
  },
};

exports['util.spawn.multibyte'] = {
  setUp: function(done) {
    this.script = path.resolve('test/fixtures/spawn-multibyte.js');
    done();
  },
  'partial stdout': function(test) {
    test.expect(4);
    util.spawn({
      cmd: process.execPath,
      args: [ this.script ],
    }, function(err, result, code) {
      test.equals(err, null);
      test.equals(code, 0);
      test.equals(result.stdout, 'こんにちは');
      test.equals(result.stderr, 'こんにちは');
      test.done();
    });
  }
};

exports['util.underscore.string'] = function(test) {
  test.expect(4);
  test.equals(util._.trim('    foo     '), 'foo', 'Should have trimmed the string.');
  test.equals(util._.capitalize('foo'), 'Foo', 'Should have capitalized the first letter.');
  test.equals(util._.words('one two three').length, 3, 'Should have counted three words.');
  test.ok(util._.isBlank(' '), 'Should be blank.');
  test.done();
};

function getType(val) {
  if (Buffer.isBuffer(val)) { return 'buffer'; }
  return Object.prototype.toString.call(val).slice(8, -1).toLowerCase();
}

exports['util.recurse'] = {
  setUp: function(done) {
    this.typeValue = function(value) {
      return {
        value: value,
        type: getType(value),
      };
    };
    done();
  },
  'primitives': function(test) {
    test.expect(1);
    var actual = util.recurse({
      bool: true,
      num: 1,
      str: 'foo',
      nul: null,
      undef: undefined,
    }, this.typeValue);
    var expected = {
      bool: {type: 'boolean', value: true},
      num: {type: 'number', value: 1},
      str: {type: 'string', value: 'foo'},
      nul: {type: 'null', value: null},
      undef: {type: 'undefined', value: undefined},
    };
    test.deepEqual(actual, expected, 'Should process primitive values.');
    test.done();
  },
  'array': function(test) {
    test.expect(1);
    var actual = util.recurse({
      arr: [
        true,
        1,
        'foo',
        null,
        undefined,
        [
          true,
          1,
          'foo',
          null,
          undefined,
        ],
      ],
    }, this.typeValue);
    var expected = {
      arr: [
        {type: 'boolean', value: true},
        {type: 'number', value: 1},
        {type: 'string', value: 'foo'},
        {type: 'null', value: null},
        {type: 'undefined', value: undefined},
        [
          {type: 'boolean', value: true},
          {type: 'number', value: 1},
          {type: 'string', value: 'foo'},
          {type: 'null', value: null},
          {type: 'undefined', value: undefined},
        ],
      ],
    };
    test.deepEqual(actual, expected, 'Should recurse over arrays.');
    test.done();
  },
  'object': function(test) {
    test.expect(1);
    var actual = util.recurse({
      obj: {
        bool: true,
        num: 1,
        str: 'foo',
        nul: null,
        undef: undefined,
        obj: {
          bool: true,
          num: 1,
          str: 'foo',
          nul: null,
          undef: undefined,
        },
      },
    }, this.typeValue);
    var expected = {
      obj: {
        bool: {type: 'boolean', value: true},
        num: {type: 'number', value: 1},
        str: {type: 'string', value: 'foo'},
        nul: {type: 'null', value: null},
        undef: {type: 'undefined', value: undefined},
        obj: {
          bool: {type: 'boolean', value: true},
          num: {type: 'number', value: 1},
          str: {type: 'string', value: 'foo'},
          nul: {type: 'null', value: null},
          undef: {type: 'undefined', value: undefined},
        },
      },
    };
    test.deepEqual(actual, expected, 'Should recurse over objects.');
    test.done();
  },
  'array in object': function(test) {
    test.expect(1);
    var actual = util.recurse({
      obj: {
        arr: [
          true,
          1,
          'foo',
          null,
          undefined,
        ],
      },
    }, this.typeValue);
    var expected = {
      obj: {
        arr: [
          {type: 'boolean', value: true},
          {type: 'number', value: 1},
          {type: 'string', value: 'foo'},
          {type: 'null', value: null},
          {type: 'undefined', value: undefined},
        ],
      },
    };
    test.deepEqual(actual, expected, 'Should recurse over arrays in objects.');
    test.done();
  },
  'object in array': function(test) {
    test.expect(1);
    var actual = util.recurse({
      arr: [
        true,
        {
          num: 1,
          str: 'foo',
        },
        null,
        undefined,
      ],
    }, this.typeValue);
    var expected = {
      arr: [
        {type: 'boolean', value: true},
        {
          num: {type: 'number', value: 1},
          str: {type: 'string', value: 'foo'},
        },
        {type: 'null', value: null},
        {type: 'undefined', value: undefined},
      ],
    };
    test.deepEqual(actual, expected, 'Should recurse over objects in arrays.');
    test.done();
  },
  'buffer': function(test) {
    test.expect(1);
    var actual = util.recurse({
      buf: new Buffer('buf'),
    }, this.typeValue);
    var expected = {
      buf: {type: 'buffer', value: new Buffer('buf')},
    };
    test.deepEqual(actual, expected, 'Should not mangle Buffer instances.');
    test.done();
  },
  'inherited properties': function(test) {
    test.expect(1);
    var actual = util.recurse({
      obj: Object.create({num: 1}, {
        str: {value: 'foo', enumerable: true},
        ignored: {value: 'ignored', enumerable: false},
      }),
    }, this.typeValue);
    var expected = {
      obj: {
        num: {type: 'number', value: 1},
        str: {type: 'string', value: 'foo'},
      }
    };
    test.deepEqual(actual, expected, 'Should enumerate inherited object properties.');
    test.done();
  },
  'circular references': function(test) {
    test.expect(6);
    function assertErrorWithPath(expectedPath) {
      return function(actual) {
        return actual.path === expectedPath &&
          actual.message === 'Circular reference detected (' + expectedPath + ')';
      };
    }
    test.doesNotThrow(function() {
      var obj = {
        // wat
        a:[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]],
        // does
        b:[[[[],[[[],[[[[],[[[],[[[],[[[],[[[],[[[[],[[]]]]]]]]]]]]]]]]]]]]],
        // it
        c:{d:{e:{f:{g:{h:{i:{j:{k:{l:{m:{n:{o:{p:{q:{r:{s:{}}}}}}}}}}}}}}}}},
        // mean
        t:[{u:[{v:[[[[],[[[],[[[{w:[{x:[[[],[[[{y:[[1]]}]]]]]}]}]]]]]]]]}]}],
      };
      util.recurse(obj, function(v) { return v; });
    }, 'Should not throw when no circular reference is detected.');
    test.throws(function() {
      var obj = {a: 1, b: 2};
      obj.obj = obj;
      util.recurse(obj, function(v) { return v; });
    }, assertErrorWithPath('.obj'), 'Should throw when a circular reference is detected.');
    test.throws(function() {
      var obj = {a:{'b b':{'c-c':{d_d:{e:{f:{g:{h:{i:{j:{k:{l:{}}}}}}}}}}}}};
      obj.a['b b']['c-c'].d_d.e.f.g.h.i.j.k.l.obj = obj;
      util.recurse(obj, function(v) { return v; });
    }, assertErrorWithPath('.a["b b"]["c-c"].d_d.e.f.g.h.i.j.k.l.obj'), 'Should throw when a circular reference is detected.');
    test.throws(function() {
      var obj = {a: 1, b: 2};
      obj.arr = [1, 2, obj, 3, 4];
      util.recurse(obj, function(v) { return v; });
    }, assertErrorWithPath('.arr[2]'), 'Should throw when a circular reference is detected.');
    test.throws(function() {
      var obj = {a: 1, b: 2};
      obj.arr = [{a:[1,{b:[2,{c:[3,obj,4]},5]},6]},7];
      util.recurse(obj, function(v) { return v; });
    }, assertErrorWithPath('.arr[0].a[1].b[1].c[1]'), 'Should throw when a circular reference is detected.');
    test.throws(function() {
      var obj = {a: 1, b: 2};
      obj.arr = [];
      obj.arr.push(0,{a:[1,{b:[2,{c:[3,obj.arr,4]},5]},6]},7);
      util.recurse(obj, function(v) { return v; });
    }, assertErrorWithPath('.arr[1].a[1].b[1].c[1]'), 'Should throw when a circular reference is detected.');
    test.done();
  },
};
