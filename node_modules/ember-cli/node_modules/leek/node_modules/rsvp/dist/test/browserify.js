(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({"./test/index.js":[function(require,module,exports){
require('./test-adapter.js');
require('./extension-test.js');
require('promises-aplus-tests-phantom/lib/testFiles');

},{"./extension-test.js":"/Users/stanley/code/rsvp.js/tmp/tree_merger-tmp_dest_dir-l9APRzgC.tmp/test/extension-test.js","./test-adapter.js":"/Users/stanley/code/rsvp.js/tmp/tree_merger-tmp_dest_dir-l9APRzgC.tmp/test/test-adapter.js","promises-aplus-tests-phantom/lib/testFiles":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/testFiles.js"}],"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js":[function(require,module,exports){
// http://wiki.commonjs.org/wiki/Unit_Testing/1.0
//
// THIS IS NOT TESTED NOR LIKELY TO WORK OUTSIDE V8!
//
// Originally from narwhal.js (http://narwhaljs.org)
// Copyright (c) 2009 Thomas Robinson <280north.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the 'Software'), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// when used in node, this will actually load the util module we depend on
// versus loading the builtin util module as happens otherwise
// this is a bug in node module loading as far as I am concerned
var util = require('util/');

var pSlice = Array.prototype.slice;
var hasOwn = Object.prototype.hasOwnProperty;

// 1. The assert module provides functions that throw
// AssertionError's when particular conditions are not met. The
// assert module must conform to the following interface.

var assert = module.exports = ok;

// 2. The AssertionError is defined in assert.
// new assert.AssertionError({ message: message,
//                             actual: actual,
//                             expected: expected })

assert.AssertionError = function AssertionError(options) {
  this.name = 'AssertionError';
  this.actual = options.actual;
  this.expected = options.expected;
  this.operator = options.operator;
  if (options.message) {
    this.message = options.message;
    this.generatedMessage = false;
  } else {
    this.message = getMessage(this);
    this.generatedMessage = true;
  }
  var stackStartFunction = options.stackStartFunction || fail;

  if (Error.captureStackTrace) {
    Error.captureStackTrace(this, stackStartFunction);
  }
  else {
    // non v8 browsers so we can have a stacktrace
    var err = new Error();
    if (err.stack) {
      var out = err.stack;

      // try to strip useless frames
      var fn_name = stackStartFunction.name;
      var idx = out.indexOf('\n' + fn_name);
      if (idx >= 0) {
        // once we have located the function frame
        // we need to strip out everything before it (and its line)
        var next_line = out.indexOf('\n', idx + 1);
        out = out.substring(next_line + 1);
      }

      this.stack = out;
    }
  }
};

// assert.AssertionError instanceof Error
util.inherits(assert.AssertionError, Error);

function replacer(key, value) {
  if (util.isUndefined(value)) {
    return '' + value;
  }
  if (util.isNumber(value) && (isNaN(value) || !isFinite(value))) {
    return value.toString();
  }
  if (util.isFunction(value) || util.isRegExp(value)) {
    return value.toString();
  }
  return value;
}

function truncate(s, n) {
  if (util.isString(s)) {
    return s.length < n ? s : s.slice(0, n);
  } else {
    return s;
  }
}

function getMessage(self) {
  return truncate(JSON.stringify(self.actual, replacer), 128) + ' ' +
         self.operator + ' ' +
         truncate(JSON.stringify(self.expected, replacer), 128);
}

// At present only the three keys mentioned above are used and
// understood by the spec. Implementations or sub modules can pass
// other keys to the AssertionError's constructor - they will be
// ignored.

// 3. All of the following functions must throw an AssertionError
// when a corresponding condition is not met, with a message that
// may be undefined if not provided.  All assertion methods provide
// both the actual and expected values to the assertion error for
// display purposes.

function fail(actual, expected, message, operator, stackStartFunction) {
  throw new assert.AssertionError({
    message: message,
    actual: actual,
    expected: expected,
    operator: operator,
    stackStartFunction: stackStartFunction
  });
}

// EXTENSION! allows for well behaved errors defined elsewhere.
assert.fail = fail;

// 4. Pure assertion tests whether a value is truthy, as determined
// by !!guard.
// assert.ok(guard, message_opt);
// This statement is equivalent to assert.equal(true, !!guard,
// message_opt);. To test strictly for the value true, use
// assert.strictEqual(true, guard, message_opt);.

function ok(value, message) {
  if (!value) fail(value, true, message, '==', assert.ok);
}
assert.ok = ok;

// 5. The equality assertion tests shallow, coercive equality with
// ==.
// assert.equal(actual, expected, message_opt);

assert.equal = function equal(actual, expected, message) {
  if (actual != expected) fail(actual, expected, message, '==', assert.equal);
};

// 6. The non-equality assertion tests for whether two objects are not equal
// with != assert.notEqual(actual, expected, message_opt);

assert.notEqual = function notEqual(actual, expected, message) {
  if (actual == expected) {
    fail(actual, expected, message, '!=', assert.notEqual);
  }
};

// 7. The equivalence assertion tests a deep equality relation.
// assert.deepEqual(actual, expected, message_opt);

assert.deepEqual = function deepEqual(actual, expected, message) {
  if (!_deepEqual(actual, expected)) {
    fail(actual, expected, message, 'deepEqual', assert.deepEqual);
  }
};

function _deepEqual(actual, expected) {
  // 7.1. All identical values are equivalent, as determined by ===.
  if (actual === expected) {
    return true;

  } else if (util.isBuffer(actual) && util.isBuffer(expected)) {
    if (actual.length != expected.length) return false;

    for (var i = 0; i < actual.length; i++) {
      if (actual[i] !== expected[i]) return false;
    }

    return true;

  // 7.2. If the expected value is a Date object, the actual value is
  // equivalent if it is also a Date object that refers to the same time.
  } else if (util.isDate(actual) && util.isDate(expected)) {
    return actual.getTime() === expected.getTime();

  // 7.3 If the expected value is a RegExp object, the actual value is
  // equivalent if it is also a RegExp object with the same source and
  // properties (`global`, `multiline`, `lastIndex`, `ignoreCase`).
  } else if (util.isRegExp(actual) && util.isRegExp(expected)) {
    return actual.source === expected.source &&
           actual.global === expected.global &&
           actual.multiline === expected.multiline &&
           actual.lastIndex === expected.lastIndex &&
           actual.ignoreCase === expected.ignoreCase;

  // 7.4. Other pairs that do not both pass typeof value == 'object',
  // equivalence is determined by ==.
  } else if (!util.isObject(actual) && !util.isObject(expected)) {
    return actual == expected;

  // 7.5 For all other Object pairs, including Array objects, equivalence is
  // determined by having the same number of owned properties (as verified
  // with Object.prototype.hasOwnProperty.call), the same set of keys
  // (although not necessarily the same order), equivalent values for every
  // corresponding key, and an identical 'prototype' property. Note: this
  // accounts for both named and indexed properties on Arrays.
  } else {
    return objEquiv(actual, expected);
  }
}

function isArguments(object) {
  return Object.prototype.toString.call(object) == '[object Arguments]';
}

function objEquiv(a, b) {
  if (util.isNullOrUndefined(a) || util.isNullOrUndefined(b))
    return false;
  // an identical 'prototype' property.
  if (a.prototype !== b.prototype) return false;
  //~~~I've managed to break Object.keys through screwy arguments passing.
  //   Converting to array solves the problem.
  if (isArguments(a)) {
    if (!isArguments(b)) {
      return false;
    }
    a = pSlice.call(a);
    b = pSlice.call(b);
    return _deepEqual(a, b);
  }
  try {
    var ka = objectKeys(a),
        kb = objectKeys(b),
        key, i;
  } catch (e) {//happens when one is a string literal and the other isn't
    return false;
  }
  // having the same number of owned properties (keys incorporates
  // hasOwnProperty)
  if (ka.length != kb.length)
    return false;
  //the same set of keys (although not necessarily the same order),
  ka.sort();
  kb.sort();
  //~~~cheap key test
  for (i = ka.length - 1; i >= 0; i--) {
    if (ka[i] != kb[i])
      return false;
  }
  //equivalent values for every corresponding key, and
  //~~~possibly expensive deep test
  for (i = ka.length - 1; i >= 0; i--) {
    key = ka[i];
    if (!_deepEqual(a[key], b[key])) return false;
  }
  return true;
}

// 8. The non-equivalence assertion tests for any deep inequality.
// assert.notDeepEqual(actual, expected, message_opt);

assert.notDeepEqual = function notDeepEqual(actual, expected, message) {
  if (_deepEqual(actual, expected)) {
    fail(actual, expected, message, 'notDeepEqual', assert.notDeepEqual);
  }
};

// 9. The strict equality assertion tests strict equality, as determined by ===.
// assert.strictEqual(actual, expected, message_opt);

assert.strictEqual = function strictEqual(actual, expected, message) {
  if (actual !== expected) {
    fail(actual, expected, message, '===', assert.strictEqual);
  }
};

// 10. The strict non-equality assertion tests for strict inequality, as
// determined by !==.  assert.notStrictEqual(actual, expected, message_opt);

assert.notStrictEqual = function notStrictEqual(actual, expected, message) {
  if (actual === expected) {
    fail(actual, expected, message, '!==', assert.notStrictEqual);
  }
};

function expectedException(actual, expected) {
  if (!actual || !expected) {
    return false;
  }

  if (Object.prototype.toString.call(expected) == '[object RegExp]') {
    return expected.test(actual);
  } else if (actual instanceof expected) {
    return true;
  } else if (expected.call({}, actual) === true) {
    return true;
  }

  return false;
}

function _throws(shouldThrow, block, expected, message) {
  var actual;

  if (util.isString(expected)) {
    message = expected;
    expected = null;
  }

  try {
    block();
  } catch (e) {
    actual = e;
  }

  message = (expected && expected.name ? ' (' + expected.name + ').' : '.') +
            (message ? ' ' + message : '.');

  if (shouldThrow && !actual) {
    fail(actual, expected, 'Missing expected exception' + message);
  }

  if (!shouldThrow && expectedException(actual, expected)) {
    fail(actual, expected, 'Got unwanted exception' + message);
  }

  if ((shouldThrow && actual && expected &&
      !expectedException(actual, expected)) || (!shouldThrow && actual)) {
    throw actual;
  }
}

// 11. Expected to throw an error:
// assert.throws(block, Error_opt, message_opt);

assert.throws = function(block, /*optional*/error, /*optional*/message) {
  _throws.apply(this, [true].concat(pSlice.call(arguments)));
};

// EXTENSION! This is annoying to write outside this module.
assert.doesNotThrow = function(block, /*optional*/message) {
  _throws.apply(this, [false].concat(pSlice.call(arguments)));
};

assert.ifError = function(err) { if (err) {throw err;}};

var objectKeys = Object.keys || function (obj) {
  var keys = [];
  for (var key in obj) {
    if (hasOwn.call(obj, key)) keys.push(key);
  }
  return keys;
};

},{"util/":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/util/util.js"}],"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/inherits/inherits_browser.js":[function(require,module,exports){
if (typeof Object.create === 'function') {
  // implementation from standard node.js 'util' module
  module.exports = function inherits(ctor, superCtor) {
    ctor.super_ = superCtor
    ctor.prototype = Object.create(superCtor.prototype, {
      constructor: {
        value: ctor,
        enumerable: false,
        writable: true,
        configurable: true
      }
    });
  };
} else {
  // old school shim for old browsers
  module.exports = function inherits(ctor, superCtor) {
    ctor.super_ = superCtor
    var TempCtor = function () {}
    TempCtor.prototype = superCtor.prototype
    ctor.prototype = new TempCtor()
    ctor.prototype.constructor = ctor
  }
}

},{}],"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/process/browser.js":[function(require,module,exports){
// shim for using process in browser

var process = module.exports = {};

process.nextTick = (function () {
    var canSetImmediate = typeof window !== 'undefined'
    && window.setImmediate;
    var canMutationObserver = typeof window !== 'undefined'
    && window.MutationObserver;
    var canPost = typeof window !== 'undefined'
    && window.postMessage && window.addEventListener
    ;

    if (canSetImmediate) {
        return function (f) { return window.setImmediate(f) };
    }

    var queue = [];

    if (canMutationObserver) {
        var hiddenDiv = document.createElement("div");
        var observer = new MutationObserver(function () {
            var queueList = queue.slice();
            queue.length = 0;
            queueList.forEach(function (fn) {
                fn();
            });
        });

        observer.observe(hiddenDiv, { attributes: true });

        return function nextTick(fn) {
            if (!queue.length) {
                hiddenDiv.setAttribute('yes', 'no');
            }
            queue.push(fn);
        };
    }

    if (canPost) {
        window.addEventListener('message', function (ev) {
            var source = ev.source;
            if ((source === window || source === null) && ev.data === 'process-tick') {
                ev.stopPropagation();
                if (queue.length > 0) {
                    var fn = queue.shift();
                    fn();
                }
            }
        }, true);

        return function nextTick(fn) {
            queue.push(fn);
            window.postMessage('process-tick', '*');
        };
    }

    return function nextTick(fn) {
        setTimeout(fn, 0);
    };
})();

process.title = 'browser';
process.browser = true;
process.env = {};
process.argv = [];

function noop() {}

process.on = noop;
process.addListener = noop;
process.once = noop;
process.off = noop;
process.removeListener = noop;
process.removeAllListeners = noop;
process.emit = noop;

process.binding = function (name) {
    throw new Error('process.binding is not supported');
};

// TODO(shtylman)
process.cwd = function () { return '/' };
process.chdir = function (dir) {
    throw new Error('process.chdir is not supported');
};

},{}],"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/util/support/isBufferBrowser.js":[function(require,module,exports){
module.exports = function isBuffer(arg) {
  return arg && typeof arg === 'object'
    && typeof arg.copy === 'function'
    && typeof arg.fill === 'function'
    && typeof arg.readUInt8 === 'function';
}
},{}],"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/util/util.js":[function(require,module,exports){
(function (process,global){
// Copyright Joyent, Inc. and other Node contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

var formatRegExp = /%[sdj%]/g;
exports.format = function(f) {
  if (!isString(f)) {
    var objects = [];
    for (var i = 0; i < arguments.length; i++) {
      objects.push(inspect(arguments[i]));
    }
    return objects.join(' ');
  }

  var i = 1;
  var args = arguments;
  var len = args.length;
  var str = String(f).replace(formatRegExp, function(x) {
    if (x === '%%') return '%';
    if (i >= len) return x;
    switch (x) {
      case '%s': return String(args[i++]);
      case '%d': return Number(args[i++]);
      case '%j':
        try {
          return JSON.stringify(args[i++]);
        } catch (_) {
          return '[Circular]';
        }
      default:
        return x;
    }
  });
  for (var x = args[i]; i < len; x = args[++i]) {
    if (isNull(x) || !isObject(x)) {
      str += ' ' + x;
    } else {
      str += ' ' + inspect(x);
    }
  }
  return str;
};


// Mark that a method should not be used.
// Returns a modified function which warns once by default.
// If --no-deprecation is set, then it is a no-op.
exports.deprecate = function(fn, msg) {
  // Allow for deprecating things in the process of starting up.
  if (isUndefined(global.process)) {
    return function() {
      return exports.deprecate(fn, msg).apply(this, arguments);
    };
  }

  if (process.noDeprecation === true) {
    return fn;
  }

  var warned = false;
  function deprecated() {
    if (!warned) {
      if (process.throwDeprecation) {
        throw new Error(msg);
      } else if (process.traceDeprecation) {
        console.trace(msg);
      } else {
        console.error(msg);
      }
      warned = true;
    }
    return fn.apply(this, arguments);
  }

  return deprecated;
};


var debugs = {};
var debugEnviron;
exports.debuglog = function(set) {
  if (isUndefined(debugEnviron))
    debugEnviron = process.env.NODE_DEBUG || '';
  set = set.toUpperCase();
  if (!debugs[set]) {
    if (new RegExp('\\b' + set + '\\b', 'i').test(debugEnviron)) {
      var pid = process.pid;
      debugs[set] = function() {
        var msg = exports.format.apply(exports, arguments);
        console.error('%s %d: %s', set, pid, msg);
      };
    } else {
      debugs[set] = function() {};
    }
  }
  return debugs[set];
};


/**
 * Echos the value of a value. Trys to print the value out
 * in the best way possible given the different types.
 *
 * @param {Object} obj The object to print out.
 * @param {Object} opts Optional options object that alters the output.
 */
/* legacy: obj, showHidden, depth, colors*/
function inspect(obj, opts) {
  // default options
  var ctx = {
    seen: [],
    stylize: stylizeNoColor
  };
  // legacy...
  if (arguments.length >= 3) ctx.depth = arguments[2];
  if (arguments.length >= 4) ctx.colors = arguments[3];
  if (isBoolean(opts)) {
    // legacy...
    ctx.showHidden = opts;
  } else if (opts) {
    // got an "options" object
    exports._extend(ctx, opts);
  }
  // set default options
  if (isUndefined(ctx.showHidden)) ctx.showHidden = false;
  if (isUndefined(ctx.depth)) ctx.depth = 2;
  if (isUndefined(ctx.colors)) ctx.colors = false;
  if (isUndefined(ctx.customInspect)) ctx.customInspect = true;
  if (ctx.colors) ctx.stylize = stylizeWithColor;
  return formatValue(ctx, obj, ctx.depth);
}
exports.inspect = inspect;


// http://en.wikipedia.org/wiki/ANSI_escape_code#graphics
inspect.colors = {
  'bold' : [1, 22],
  'italic' : [3, 23],
  'underline' : [4, 24],
  'inverse' : [7, 27],
  'white' : [37, 39],
  'grey' : [90, 39],
  'black' : [30, 39],
  'blue' : [34, 39],
  'cyan' : [36, 39],
  'green' : [32, 39],
  'magenta' : [35, 39],
  'red' : [31, 39],
  'yellow' : [33, 39]
};

// Don't use 'blue' not visible on cmd.exe
inspect.styles = {
  'special': 'cyan',
  'number': 'yellow',
  'boolean': 'yellow',
  'undefined': 'grey',
  'null': 'bold',
  'string': 'green',
  'date': 'magenta',
  // "name": intentionally not styling
  'regexp': 'red'
};


function stylizeWithColor(str, styleType) {
  var style = inspect.styles[styleType];

  if (style) {
    return '\u001b[' + inspect.colors[style][0] + 'm' + str +
           '\u001b[' + inspect.colors[style][1] + 'm';
  } else {
    return str;
  }
}


function stylizeNoColor(str, styleType) {
  return str;
}


function arrayToHash(array) {
  var hash = {};

  array.forEach(function(val, idx) {
    hash[val] = true;
  });

  return hash;
}


function formatValue(ctx, value, recurseTimes) {
  // Provide a hook for user-specified inspect functions.
  // Check that value is an object with an inspect function on it
  if (ctx.customInspect &&
      value &&
      isFunction(value.inspect) &&
      // Filter out the util module, it's inspect function is special
      value.inspect !== exports.inspect &&
      // Also filter out any prototype objects using the circular check.
      !(value.constructor && value.constructor.prototype === value)) {
    var ret = value.inspect(recurseTimes, ctx);
    if (!isString(ret)) {
      ret = formatValue(ctx, ret, recurseTimes);
    }
    return ret;
  }

  // Primitive types cannot have properties
  var primitive = formatPrimitive(ctx, value);
  if (primitive) {
    return primitive;
  }

  // Look up the keys of the object.
  var keys = Object.keys(value);
  var visibleKeys = arrayToHash(keys);

  if (ctx.showHidden) {
    keys = Object.getOwnPropertyNames(value);
  }

  // IE doesn't make error fields non-enumerable
  // http://msdn.microsoft.com/en-us/library/ie/dww52sbt(v=vs.94).aspx
  if (isError(value)
      && (keys.indexOf('message') >= 0 || keys.indexOf('description') >= 0)) {
    return formatError(value);
  }

  // Some type of object without properties can be shortcutted.
  if (keys.length === 0) {
    if (isFunction(value)) {
      var name = value.name ? ': ' + value.name : '';
      return ctx.stylize('[Function' + name + ']', 'special');
    }
    if (isRegExp(value)) {
      return ctx.stylize(RegExp.prototype.toString.call(value), 'regexp');
    }
    if (isDate(value)) {
      return ctx.stylize(Date.prototype.toString.call(value), 'date');
    }
    if (isError(value)) {
      return formatError(value);
    }
  }

  var base = '', array = false, braces = ['{', '}'];

  // Make Array say that they are Array
  if (isArray(value)) {
    array = true;
    braces = ['[', ']'];
  }

  // Make functions say that they are functions
  if (isFunction(value)) {
    var n = value.name ? ': ' + value.name : '';
    base = ' [Function' + n + ']';
  }

  // Make RegExps say that they are RegExps
  if (isRegExp(value)) {
    base = ' ' + RegExp.prototype.toString.call(value);
  }

  // Make dates with properties first say the date
  if (isDate(value)) {
    base = ' ' + Date.prototype.toUTCString.call(value);
  }

  // Make error with message first say the error
  if (isError(value)) {
    base = ' ' + formatError(value);
  }

  if (keys.length === 0 && (!array || value.length == 0)) {
    return braces[0] + base + braces[1];
  }

  if (recurseTimes < 0) {
    if (isRegExp(value)) {
      return ctx.stylize(RegExp.prototype.toString.call(value), 'regexp');
    } else {
      return ctx.stylize('[Object]', 'special');
    }
  }

  ctx.seen.push(value);

  var output;
  if (array) {
    output = formatArray(ctx, value, recurseTimes, visibleKeys, keys);
  } else {
    output = keys.map(function(key) {
      return formatProperty(ctx, value, recurseTimes, visibleKeys, key, array);
    });
  }

  ctx.seen.pop();

  return reduceToSingleString(output, base, braces);
}


function formatPrimitive(ctx, value) {
  if (isUndefined(value))
    return ctx.stylize('undefined', 'undefined');
  if (isString(value)) {
    var simple = '\'' + JSON.stringify(value).replace(/^"|"$/g, '')
                                             .replace(/'/g, "\\'")
                                             .replace(/\\"/g, '"') + '\'';
    return ctx.stylize(simple, 'string');
  }
  if (isNumber(value))
    return ctx.stylize('' + value, 'number');
  if (isBoolean(value))
    return ctx.stylize('' + value, 'boolean');
  // For some reason typeof null is "object", so special case here.
  if (isNull(value))
    return ctx.stylize('null', 'null');
}


function formatError(value) {
  return '[' + Error.prototype.toString.call(value) + ']';
}


function formatArray(ctx, value, recurseTimes, visibleKeys, keys) {
  var output = [];
  for (var i = 0, l = value.length; i < l; ++i) {
    if (hasOwnProperty(value, String(i))) {
      output.push(formatProperty(ctx, value, recurseTimes, visibleKeys,
          String(i), true));
    } else {
      output.push('');
    }
  }
  keys.forEach(function(key) {
    if (!key.match(/^\d+$/)) {
      output.push(formatProperty(ctx, value, recurseTimes, visibleKeys,
          key, true));
    }
  });
  return output;
}


function formatProperty(ctx, value, recurseTimes, visibleKeys, key, array) {
  var name, str, desc;
  desc = Object.getOwnPropertyDescriptor(value, key) || { value: value[key] };
  if (desc.get) {
    if (desc.set) {
      str = ctx.stylize('[Getter/Setter]', 'special');
    } else {
      str = ctx.stylize('[Getter]', 'special');
    }
  } else {
    if (desc.set) {
      str = ctx.stylize('[Setter]', 'special');
    }
  }
  if (!hasOwnProperty(visibleKeys, key)) {
    name = '[' + key + ']';
  }
  if (!str) {
    if (ctx.seen.indexOf(desc.value) < 0) {
      if (isNull(recurseTimes)) {
        str = formatValue(ctx, desc.value, null);
      } else {
        str = formatValue(ctx, desc.value, recurseTimes - 1);
      }
      if (str.indexOf('\n') > -1) {
        if (array) {
          str = str.split('\n').map(function(line) {
            return '  ' + line;
          }).join('\n').substr(2);
        } else {
          str = '\n' + str.split('\n').map(function(line) {
            return '   ' + line;
          }).join('\n');
        }
      }
    } else {
      str = ctx.stylize('[Circular]', 'special');
    }
  }
  if (isUndefined(name)) {
    if (array && key.match(/^\d+$/)) {
      return str;
    }
    name = JSON.stringify('' + key);
    if (name.match(/^"([a-zA-Z_][a-zA-Z_0-9]*)"$/)) {
      name = name.substr(1, name.length - 2);
      name = ctx.stylize(name, 'name');
    } else {
      name = name.replace(/'/g, "\\'")
                 .replace(/\\"/g, '"')
                 .replace(/(^"|"$)/g, "'");
      name = ctx.stylize(name, 'string');
    }
  }

  return name + ': ' + str;
}


function reduceToSingleString(output, base, braces) {
  var numLinesEst = 0;
  var length = output.reduce(function(prev, cur) {
    numLinesEst++;
    if (cur.indexOf('\n') >= 0) numLinesEst++;
    return prev + cur.replace(/\u001b\[\d\d?m/g, '').length + 1;
  }, 0);

  if (length > 60) {
    return braces[0] +
           (base === '' ? '' : base + '\n ') +
           ' ' +
           output.join(',\n  ') +
           ' ' +
           braces[1];
  }

  return braces[0] + base + ' ' + output.join(', ') + ' ' + braces[1];
}


// NOTE: These type checking functions intentionally don't use `instanceof`
// because it is fragile and can be easily faked with `Object.create()`.
function isArray(ar) {
  return Array.isArray(ar);
}
exports.isArray = isArray;

function isBoolean(arg) {
  return typeof arg === 'boolean';
}
exports.isBoolean = isBoolean;

function isNull(arg) {
  return arg === null;
}
exports.isNull = isNull;

function isNullOrUndefined(arg) {
  return arg == null;
}
exports.isNullOrUndefined = isNullOrUndefined;

function isNumber(arg) {
  return typeof arg === 'number';
}
exports.isNumber = isNumber;

function isString(arg) {
  return typeof arg === 'string';
}
exports.isString = isString;

function isSymbol(arg) {
  return typeof arg === 'symbol';
}
exports.isSymbol = isSymbol;

function isUndefined(arg) {
  return arg === void 0;
}
exports.isUndefined = isUndefined;

function isRegExp(re) {
  return isObject(re) && objectToString(re) === '[object RegExp]';
}
exports.isRegExp = isRegExp;

function isObject(arg) {
  return typeof arg === 'object' && arg !== null;
}
exports.isObject = isObject;

function isDate(d) {
  return isObject(d) && objectToString(d) === '[object Date]';
}
exports.isDate = isDate;

function isError(e) {
  return isObject(e) &&
      (objectToString(e) === '[object Error]' || e instanceof Error);
}
exports.isError = isError;

function isFunction(arg) {
  return typeof arg === 'function';
}
exports.isFunction = isFunction;

function isPrimitive(arg) {
  return arg === null ||
         typeof arg === 'boolean' ||
         typeof arg === 'number' ||
         typeof arg === 'string' ||
         typeof arg === 'symbol' ||  // ES6 symbol
         typeof arg === 'undefined';
}
exports.isPrimitive = isPrimitive;

exports.isBuffer = require('./support/isBuffer');

function objectToString(o) {
  return Object.prototype.toString.call(o);
}


function pad(n) {
  return n < 10 ? '0' + n.toString(10) : n.toString(10);
}


var months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
              'Oct', 'Nov', 'Dec'];

// 26 Feb 16:19:34
function timestamp() {
  var d = new Date();
  var time = [pad(d.getHours()),
              pad(d.getMinutes()),
              pad(d.getSeconds())].join(':');
  return [d.getDate(), months[d.getMonth()], time].join(' ');
}


// log is just a thin wrapper to console.log that prepends a timestamp
exports.log = function() {
  console.log('%s - %s', timestamp(), exports.format.apply(exports, arguments));
};


/**
 * Inherit the prototype methods from one constructor into another.
 *
 * The Function.prototype.inherits from lang.js rewritten as a standalone
 * function (not on Function.prototype). NOTE: If this file is to be loaded
 * during bootstrapping this function needs to be rewritten using some native
 * functions as prototype setup using normal JavaScript does not work as
 * expected during bootstrapping (see mirror.js in r114903).
 *
 * @param {function} ctor Constructor function which needs to inherit the
 *     prototype.
 * @param {function} superCtor Constructor function to inherit prototype from.
 */
exports.inherits = require('inherits');

exports._extend = function(origin, add) {
  // Don't do anything if add isn't an object
  if (!add || !isObject(add)) return origin;

  var keys = Object.keys(add);
  var i = keys.length;
  while (i--) {
    origin[keys[i]] = add[keys[i]];
  }
  return origin;
};

function hasOwnProperty(obj, prop) {
  return Object.prototype.hasOwnProperty.call(obj, prop);
}

}).call(this,require('_process'),typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./support/isBuffer":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/util/support/isBufferBrowser.js","_process":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/process/browser.js","inherits":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/inherits/inherits_browser.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/testFiles.js":[function(require,module,exports){
require("./tests/2.1.2");
require("./tests/2.1.3");
require("./tests/2.2.1");
require("./tests/2.2.2");
require("./tests/2.2.3");
require("./tests/2.2.4");
require("./tests/2.2.5");
require("./tests/2.2.6");
require("./tests/2.2.7");
require("./tests/2.3.1");
require("./tests/2.3.2");
require("./tests/2.3.3");
require("./tests/2.3.4");

},{"./tests/2.1.2":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.1.2.js","./tests/2.1.3":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.1.3.js","./tests/2.2.1":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.1.js","./tests/2.2.2":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.2.js","./tests/2.2.3":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.3.js","./tests/2.2.4":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.4.js","./tests/2.2.5":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.5.js","./tests/2.2.6":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.6.js","./tests/2.2.7":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.7.js","./tests/2.3.1":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.3.1.js","./tests/2.3.2":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.3.2.js","./tests/2.3.3":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.3.3.js","./tests/2.3.4":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.3.4.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.1.2.js":[function(require,module,exports){
(function (global){
"use strict";

var assert = require("assert");
var testFulfilled = require("./helpers/testThreeCases").testFulfilled;

var adapter = global.adapter;
var deferred = adapter.deferred;

var dummy = { dummy: "dummy" }; // we fulfill or reject with this when we don't intend to test against it

describe("2.1.2.1: When fulfilled, a promise: must not transition to any other state.", function () {
    testFulfilled(dummy, function (promise, done) {
        var onFulfilledCalled = false;

        promise.then(function onFulfilled() {
            onFulfilledCalled = true;
        }, function onRejected() {
            assert.strictEqual(onFulfilledCalled, false);
            done();
        });

        setTimeout(done, 100);
    });

    specify("trying to fulfill then immediately reject", function (done) {
        var d = deferred();
        var onFulfilledCalled = false;

        d.promise.then(function onFulfilled() {
            onFulfilledCalled = true;
        }, function onRejected() {
            assert.strictEqual(onFulfilledCalled, false);
            done();
        });

        d.resolve(dummy);
        d.reject(dummy);
        setTimeout(done, 100);
    });

    specify("trying to fulfill then reject, delayed", function (done) {
        var d = deferred();
        var onFulfilledCalled = false;

        d.promise.then(function onFulfilled() {
            onFulfilledCalled = true;
        }, function onRejected() {
            assert.strictEqual(onFulfilledCalled, false);
            done();
        });

        setTimeout(function () {
            d.resolve(dummy);
            d.reject(dummy);
        }, 50);
        setTimeout(done, 100);
    });

    specify("trying to fulfill immediately then reject delayed", function (done) {
        var d = deferred();
        var onFulfilledCalled = false;

        d.promise.then(function onFulfilled() {
            onFulfilledCalled = true;
        }, function onRejected() {
            assert.strictEqual(onFulfilledCalled, false);
            done();
        });

        d.resolve(dummy);
        setTimeout(function () {
            d.reject(dummy);
        }, 50);
        setTimeout(done, 100);
    });
});

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./helpers/testThreeCases":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/testThreeCases.js","assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.1.3.js":[function(require,module,exports){
(function (global){
"use strict";

var assert = require("assert");
var testRejected = require("./helpers/testThreeCases").testRejected;

var adapter = global.adapter;
var deferred = adapter.deferred;

var dummy = { dummy: "dummy" }; // we fulfill or reject with this when we don't intend to test against it

describe("2.1.3.1: When rejected, a promise: must not transition to any other state.", function () {
    testRejected(dummy, function (promise, done) {
        var onRejectedCalled = false;

        promise.then(function onFulfilled() {
            assert.strictEqual(onRejectedCalled, false);
            done();
        }, function onRejected() {
            onRejectedCalled = true;
        });

        setTimeout(done, 100);
    });

    specify("trying to reject then immediately fulfill", function (done) {
        var d = deferred();
        var onRejectedCalled = false;

        d.promise.then(function onFulfilled() {
            assert.strictEqual(onRejectedCalled, false);
            done();
        }, function onRejected() {
            onRejectedCalled = true;
        });

        d.reject(dummy);
        d.resolve(dummy);
        setTimeout(done, 100);
    });

    specify("trying to reject then fulfill, delayed", function (done) {
        var d = deferred();
        var onRejectedCalled = false;

        d.promise.then(function onFulfilled() {
            assert.strictEqual(onRejectedCalled, false);
            done();
        }, function onRejected() {
            onRejectedCalled = true;
        });

        setTimeout(function () {
            d.reject(dummy);
            d.resolve(dummy);
        }, 50);
        setTimeout(done, 100);
    });

    specify("trying to reject immediately then fulfill delayed", function (done) {
        var d = deferred();
        var onRejectedCalled = false;

        d.promise.then(function onFulfilled() {
            assert.strictEqual(onRejectedCalled, false);
            done();
        }, function onRejected() {
            onRejectedCalled = true;
        });

        d.reject(dummy);
        setTimeout(function () {
            d.resolve(dummy);
        }, 50);
        setTimeout(done, 100);
    });
});

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./helpers/testThreeCases":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/testThreeCases.js","assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.1.js":[function(require,module,exports){
(function (global){
"use strict";

var adapter = global.adapter;
var resolved = adapter.resolved;
var rejected = adapter.rejected;

var dummy = { dummy: "dummy" }; // we fulfill or reject with this when we don't intend to test against it

describe("2.2.1: Both `onFulfilled` and `onRejected` are optional arguments.", function () {
    describe("2.2.1.1: If `onFulfilled` is not a function, it must be ignored.", function () {
        describe("applied to a directly-rejected promise", function () {
            function testNonFunction(nonFunction, stringRepresentation) {
                specify("`onFulfilled` is " + stringRepresentation, function (done) {
                    rejected(dummy).then(nonFunction, function () {
                        done();
                    });
                });
            }

            testNonFunction(undefined, "`undefined`");
            testNonFunction(null, "`null`");
            testNonFunction(false, "`false`");
            testNonFunction(5, "`5`");
            testNonFunction({}, "an object");
        });

        describe("applied to a promise rejected and then chained off of", function () {
            function testNonFunction(nonFunction, stringRepresentation) {
                specify("`onFulfilled` is " + stringRepresentation, function (done) {
                    rejected(dummy).then(function () { }, undefined).then(nonFunction, function () {
                        done();
                    });
                });
            }

            testNonFunction(undefined, "`undefined`");
            testNonFunction(null, "`null`");
            testNonFunction(false, "`false`");
            testNonFunction(5, "`5`");
            testNonFunction({}, "an object");
        });
    });

    describe("2.2.1.2: If `onRejected` is not a function, it must be ignored.", function () {
        describe("applied to a directly-fulfilled promise", function () {
            function testNonFunction(nonFunction, stringRepresentation) {
                specify("`onRejected` is " + stringRepresentation, function (done) {
                    resolved(dummy).then(function () {
                        done();
                    }, nonFunction);
                });
            }

            testNonFunction(undefined, "`undefined`");
            testNonFunction(null, "`null`");
            testNonFunction(false, "`false`");
            testNonFunction(5, "`5`");
            testNonFunction({}, "an object");
        });

        describe("applied to a promise fulfilled and then chained off of", function () {
            function testNonFunction(nonFunction, stringRepresentation) {
                specify("`onFulfilled` is " + stringRepresentation, function (done) {
                    resolved(dummy).then(undefined, function () { }).then(function () {
                        done();
                    }, nonFunction);
                });
            }

            testNonFunction(undefined, "`undefined`");
            testNonFunction(null, "`null`");
            testNonFunction(false, "`false`");
            testNonFunction(5, "`5`");
            testNonFunction({}, "an object");
        });
    });
});

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.2.js":[function(require,module,exports){
(function (global){
"use strict";

var assert = require("assert");
var testFulfilled = require("./helpers/testThreeCases").testFulfilled;

var adapter = global.adapter;
var resolved = adapter.resolved;
var deferred = adapter.deferred;

var dummy = { dummy: "dummy" }; // we fulfill or reject with this when we don't intend to test against it
var sentinel = { sentinel: "sentinel" }; // a sentinel fulfillment value to test for with strict equality

describe("2.2.2: If `onFulfilled` is a function,", function () {
    describe("2.2.2.1: it must be called after `promise` is fulfilled, with `promise`’s fulfillment value as its " +
             "first argument.", function () {
        testFulfilled(sentinel, function (promise, done) {
            promise.then(function onFulfilled(value) {
                assert.strictEqual(value, sentinel);
                done();
            });
        });
    });

    describe("2.2.2.2: it must not be called before `promise` is fulfilled", function () {
        specify("fulfilled after a delay", function (done) {
            var d = deferred();
            var isFulfilled = false;

            d.promise.then(function onFulfilled() {
                assert.strictEqual(isFulfilled, true);
                done();
            });

            setTimeout(function () {
                d.resolve(dummy);
                isFulfilled = true;
            }, 50);
        });

        specify("never fulfilled", function (done) {
            var d = deferred();
            var onFulfilledCalled = false;

            d.promise.then(function onFulfilled() {
                onFulfilledCalled = true;
                done();
            });

            setTimeout(function () {
                assert.strictEqual(onFulfilledCalled, false);
                done();
            }, 150);
        });
    });

    describe("2.2.2.3: it must not be called more than once.", function () {
        specify("already-fulfilled", function (done) {
            var timesCalled = 0;

            resolved(dummy).then(function onFulfilled() {
                assert.strictEqual(++timesCalled, 1);
                done();
            });
        });

        specify("trying to fulfill a pending promise more than once, immediately", function (done) {
            var d = deferred();
            var timesCalled = 0;

            d.promise.then(function onFulfilled() {
                assert.strictEqual(++timesCalled, 1);
                done();
            });

            d.resolve(dummy);
            d.resolve(dummy);
        });

        specify("trying to fulfill a pending promise more than once, delayed", function (done) {
            var d = deferred();
            var timesCalled = 0;

            d.promise.then(function onFulfilled() {
                assert.strictEqual(++timesCalled, 1);
                done();
            });

            setTimeout(function () {
                d.resolve(dummy);
                d.resolve(dummy);
            }, 50);
        });

        specify("trying to fulfill a pending promise more than once, immediately then delayed", function (done) {
            var d = deferred();
            var timesCalled = 0;

            d.promise.then(function onFulfilled() {
                assert.strictEqual(++timesCalled, 1);
                done();
            });

            d.resolve(dummy);
            setTimeout(function () {
                d.resolve(dummy);
            }, 50);
        });

        specify("when multiple `then` calls are made, spaced apart in time", function (done) {
            var d = deferred();
            var timesCalled = [0, 0, 0];

            d.promise.then(function onFulfilled() {
                assert.strictEqual(++timesCalled[0], 1);
            });

            setTimeout(function () {
                d.promise.then(function onFulfilled() {
                    assert.strictEqual(++timesCalled[1], 1);
                });
            }, 50);

            setTimeout(function () {
                d.promise.then(function onFulfilled() {
                    assert.strictEqual(++timesCalled[2], 1);
                    done();
                });
            }, 100);

            setTimeout(function () {
                d.resolve(dummy);
            }, 150);
        });

        specify("when `then` is interleaved with fulfillment", function (done) {
            var d = deferred();
            var timesCalled = [0, 0];

            d.promise.then(function onFulfilled() {
                assert.strictEqual(++timesCalled[0], 1);
            });

            d.resolve(dummy);

            d.promise.then(function onFulfilled() {
                assert.strictEqual(++timesCalled[1], 1);
                done();
            });
        });
    });
});

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./helpers/testThreeCases":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/testThreeCases.js","assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.3.js":[function(require,module,exports){
(function (global){
"use strict";

var assert = require("assert");
var testRejected = require("./helpers/testThreeCases").testRejected;

var adapter = global.adapter;
var rejected = adapter.rejected;
var deferred = adapter.deferred;

var dummy = { dummy: "dummy" }; // we fulfill or reject with this when we don't intend to test against it
var sentinel = { sentinel: "sentinel" }; // a sentinel fulfillment value to test for with strict equality

describe("2.2.3: If `onRejected` is a function,", function () {
    describe("2.2.3.1: it must be called after `promise` is rejected, with `promise`’s rejection reason as its " +
             "first argument.", function () {
        testRejected(sentinel, function (promise, done) {
            promise.then(null, function onRejected(reason) {
                assert.strictEqual(reason, sentinel);
                done();
            });
        });
    });

    describe("2.2.3.2: it must not be called before `promise` is rejected", function () {
        specify("rejected after a delay", function (done) {
            var d = deferred();
            var isRejected = false;

            d.promise.then(null, function onRejected() {
                assert.strictEqual(isRejected, true);
                done();
            });

            setTimeout(function () {
                d.reject(dummy);
                isRejected = true;
            }, 50);
        });

        specify("never rejected", function (done) {
            var d = deferred();
            var onRejectedCalled = false;

            d.promise.then(null, function onRejected() {
                onRejectedCalled = true;
                done();
            });

            setTimeout(function () {
                assert.strictEqual(onRejectedCalled, false);
                done();
            }, 150);
        });
    });

    describe("2.2.3.3: it must not be called more than once.", function () {
        specify("already-rejected", function (done) {
            var timesCalled = 0;

            rejected(dummy).then(null, function onRejected() {
                assert.strictEqual(++timesCalled, 1);
                done();
            });
        });

        specify("trying to reject a pending promise more than once, immediately", function (done) {
            var d = deferred();
            var timesCalled = 0;

            d.promise.then(null, function onRejected() {
                assert.strictEqual(++timesCalled, 1);
                done();
            });

            d.reject(dummy);
            d.reject(dummy);
        });

        specify("trying to reject a pending promise more than once, delayed", function (done) {
            var d = deferred();
            var timesCalled = 0;

            d.promise.then(null, function onRejected() {
                assert.strictEqual(++timesCalled, 1);
                done();
            });

            setTimeout(function () {
                d.reject(dummy);
                d.reject(dummy);
            }, 50);
        });

        specify("trying to reject a pending promise more than once, immediately then delayed", function (done) {
            var d = deferred();
            var timesCalled = 0;

            d.promise.then(null, function onRejected() {
                assert.strictEqual(++timesCalled, 1);
                done();
            });

            d.reject(dummy);
            setTimeout(function () {
                d.reject(dummy);
            }, 50);
        });

        specify("when multiple `then` calls are made, spaced apart in time", function (done) {
            var d = deferred();
            var timesCalled = [0, 0, 0];

            d.promise.then(null, function onRejected() {
                assert.strictEqual(++timesCalled[0], 1);
            });

            setTimeout(function () {
                d.promise.then(null, function onRejected() {
                    assert.strictEqual(++timesCalled[1], 1);
                });
            }, 50);

            setTimeout(function () {
                d.promise.then(null, function onRejected() {
                    assert.strictEqual(++timesCalled[2], 1);
                    done();
                });
            }, 100);

            setTimeout(function () {
                d.reject(dummy);
            }, 150);
        });

        specify("when `then` is interleaved with rejection", function (done) {
            var d = deferred();
            var timesCalled = [0, 0];

            d.promise.then(null, function onRejected() {
                assert.strictEqual(++timesCalled[0], 1);
            });

            d.reject(dummy);

            d.promise.then(null, function onRejected() {
                assert.strictEqual(++timesCalled[1], 1);
                done();
            });
        });
    });
});

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./helpers/testThreeCases":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/testThreeCases.js","assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.4.js":[function(require,module,exports){
(function (global){
"use strict";

var assert = require("assert");
var testFulfilled = require("./helpers/testThreeCases").testFulfilled;
var testRejected = require("./helpers/testThreeCases").testRejected;

var adapter = global.adapter;
var resolved = adapter.resolved;
var rejected = adapter.rejected;
var deferred = adapter.deferred;

var dummy = { dummy: "dummy" }; // we fulfill or reject with this when we don't intend to test against it

describe("2.2.4: `onFulfilled` or `onRejected` must not be called until the execution context stack contains only " +
         "platform code.", function () {
    describe("`then` returns before the promise becomes fulfilled or rejected", function () {
        testFulfilled(dummy, function (promise, done) {
            var thenHasReturned = false;

            promise.then(function onFulfilled() {
                assert.strictEqual(thenHasReturned, true);
                done();
            });

            thenHasReturned = true;
        });
        testRejected(dummy, function (promise, done) {
            var thenHasReturned = false;

            promise.then(null, function onRejected() {
                assert.strictEqual(thenHasReturned, true);
                done();
            });

            thenHasReturned = true;
        });
    });

    describe("Clean-stack execution ordering tests (fulfillment case)", function () {
        specify("when `onFulfilled` is added immediately before the promise is fulfilled",
                function () {
            var d = deferred();
            var onFulfilledCalled = false;

            d.promise.then(function onFulfilled() {
                onFulfilledCalled = true;
            });

            d.resolve(dummy);

            assert.strictEqual(onFulfilledCalled, false);
        });

        specify("when `onFulfilled` is added immediately after the promise is fulfilled",
                function () {
            var d = deferred();
            var onFulfilledCalled = false;

            d.resolve(dummy);

            d.promise.then(function onFulfilled() {
                onFulfilledCalled = true;
            });

            assert.strictEqual(onFulfilledCalled, false);
        });

        specify("when one `onFulfilled` is added inside another `onFulfilled`", function (done) {
            var promise = resolved();
            var firstOnFulfilledFinished = false;

            promise.then(function () {
                promise.then(function () {
                    assert.strictEqual(firstOnFulfilledFinished, true);
                    done();
                });
                firstOnFulfilledFinished = true;
            });
        });

        specify("when `onFulfilled` is added inside an `onRejected`", function (done) {
            var promise = rejected();
            var promise2 = resolved();
            var firstOnRejectedFinished = false;

            promise.then(null, function () {
                promise2.then(function () {
                    assert.strictEqual(firstOnRejectedFinished, true);
                    done();
                });
                firstOnRejectedFinished = true;
            });
        });

        specify("when the promise is fulfilled asynchronously", function (done) {
            var d = deferred();
            var firstStackFinished = false;

            setTimeout(function () {
                d.resolve(dummy);
                firstStackFinished = true;
            }, 0);

            d.promise.then(function () {
                assert.strictEqual(firstStackFinished, true);
                done();
            });
        });
    });

    describe("Clean-stack execution ordering tests (rejection case)", function () {
        specify("when `onRejected` is added immediately before the promise is rejected",
                function () {
            var d = deferred();
            var onRejectedCalled = false;

            d.promise.then(null, function onRejected() {
                onRejectedCalled = true;
            });

            d.reject(dummy);

            assert.strictEqual(onRejectedCalled, false);
        });

        specify("when `onRejected` is added immediately after the promise is rejected",
                function () {
            var d = deferred();
            var onRejectedCalled = false;

            d.reject(dummy);

            d.promise.then(null, function onRejected() {
                onRejectedCalled = true;
            });

            assert.strictEqual(onRejectedCalled, false);
        });

        specify("when `onRejected` is added inside an `onFulfilled`", function (done) {
            var promise = resolved();
            var promise2 = rejected();
            var firstOnFulfilledFinished = false;

            promise.then(function () {
                promise2.then(null, function () {
                    assert.strictEqual(firstOnFulfilledFinished, true);
                    done();
                });
                firstOnFulfilledFinished = true;
            });
        });

        specify("when one `onRejected` is added inside another `onRejected`", function (done) {
            var promise = rejected();
            var firstOnRejectedFinished = false;

            promise.then(null, function () {
                promise.then(null, function () {
                    assert.strictEqual(firstOnRejectedFinished, true);
                    done();
                });
                firstOnRejectedFinished = true;
            });
        });

        specify("when the promise is rejected asynchronously", function (done) {
            var d = deferred();
            var firstStackFinished = false;

            setTimeout(function () {
                d.reject(dummy);
                firstStackFinished = true;
            }, 0);

            d.promise.then(null, function () {
                assert.strictEqual(firstStackFinished, true);
                done();
            });
        });
    });
});

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./helpers/testThreeCases":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/testThreeCases.js","assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.5.js":[function(require,module,exports){
(function (global){
/*jshint strict: false */

var assert = require("assert");

var adapter = global.adapter;
var resolved = adapter.resolved;
var rejected = adapter.rejected;

var dummy = { dummy: "dummy" }; // we fulfill or reject with this when we don't intend to test against it
function impimentsUseStrictCorrectly() {
    "use strict";
    function test() {
        /*jshint validthis:true */
        return !this;
    }
    return test();
}
describe("2.2.5 `onFulfilled` and `onRejected` must be called as functions (i.e. with no `this` value).", function () {
    if (impimentsUseStrictCorrectly()) {
        describe("strict mode", function () {
            specify("fulfilled", function (done) {
                resolved(dummy).then(function onFulfilled() {
                    "use strict";

                    assert.strictEqual(this, undefined);
                    done();
                });
            });

            specify("rejected", function (done) {
                rejected(dummy).then(null, function onRejected() {
                    "use strict";

                    assert.strictEqual(this, undefined);
                    done();
                });
            });
        });
    }
    describe("sloppy mode", function () {
        specify("fulfilled", function (done) {
            resolved(dummy).then(function onFulfilled() {
                assert.strictEqual(this, global);
                done();
            });
        });

        specify("rejected", function (done) {
            rejected(dummy).then(null, function onRejected() {
                assert.strictEqual(this, global);
                done();
            });
        });
    });
});

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.6.js":[function(require,module,exports){
"use strict";

var assert = require("assert");
var sinon = require("sinon");
var testFulfilled = require("./helpers/testThreeCases").testFulfilled;
var testRejected = require("./helpers/testThreeCases").testRejected;

var dummy = { dummy: "dummy" }; // we fulfill or reject with this when we don't intend to test against it
var other = { other: "other" }; // a value we don't want to be strict equal to
var sentinel = { sentinel: "sentinel" }; // a sentinel fulfillment value to test for with strict equality
var sentinel2 = { sentinel2: "sentinel2" };
var sentinel3 = { sentinel3: "sentinel3" };

function callbackAggregator(times, ultimateCallback) {
    var soFar = 0;
    return function () {
        if (++soFar === times) {
            ultimateCallback();
        }
    };
}

describe("2.2.6: `then` may be called multiple times on the same promise.", function () {
    describe("2.2.6.1: If/when `promise` is fulfilled, all respective `onFulfilled` callbacks must execute in the " +
             "order of their originating calls to `then`.", function () {
        describe("multiple boring fulfillment handlers", function () {
            testFulfilled(sentinel, function (promise, done) {
                var handler1 = sinon.stub().returns(other);
                var handler2 = sinon.stub().returns(other);
                var handler3 = sinon.stub().returns(other);

                var spy = sinon.spy();
                promise.then(handler1, spy);
                promise.then(handler2, spy);
                promise.then(handler3, spy);

                promise.then(function (value) {
                    assert.strictEqual(value, sentinel);

                    sinon.assert.calledWith(handler1, sinon.match.same(sentinel));
                    sinon.assert.calledWith(handler2, sinon.match.same(sentinel));
                    sinon.assert.calledWith(handler3, sinon.match.same(sentinel));
                    sinon.assert.notCalled(spy);

                    done();
                });
            });
        });

        describe("multiple fulfillment handlers, one of which throws", function () {
            testFulfilled(sentinel, function (promise, done) {
                var handler1 = sinon.stub().returns(other);
                var handler2 = sinon.stub().throws(other);
                var handler3 = sinon.stub().returns(other);

                var spy = sinon.spy();
                promise.then(handler1, spy);
                promise.then(handler2, spy);
                promise.then(handler3, spy);

                promise.then(function (value) {
                    assert.strictEqual(value, sentinel);

                    sinon.assert.calledWith(handler1, sinon.match.same(sentinel));
                    sinon.assert.calledWith(handler2, sinon.match.same(sentinel));
                    sinon.assert.calledWith(handler3, sinon.match.same(sentinel));
                    sinon.assert.notCalled(spy);

                    done();
                });
            });
        });

        describe("results in multiple branching chains with their own fulfillment values", function () {
            testFulfilled(dummy, function (promise, done) {
                var semiDone = callbackAggregator(3, done);

                promise.then(function () {
                    return sentinel;
                }).then(function (value) {
                    assert.strictEqual(value, sentinel);
                    semiDone();
                });

                promise.then(function () {
                    throw sentinel2;
                }).then(null, function (reason) {
                    assert.strictEqual(reason, sentinel2);
                    semiDone();
                });

                promise.then(function () {
                    return sentinel3;
                }).then(function (value) {
                    assert.strictEqual(value, sentinel3);
                    semiDone();
                });
            });
        });

        describe("`onFulfilled` handlers are called in the original order", function () {
            testFulfilled(dummy, function (promise, done) {
                var handler1 = sinon.spy(function handler1() {});
                var handler2 = sinon.spy(function handler2() {});
                var handler3 = sinon.spy(function handler3() {});

                promise.then(handler1);
                promise.then(handler2);
                promise.then(handler3);

                promise.then(function () {
                    sinon.assert.callOrder(handler1, handler2, handler3);
                    done();
                });
            });

            describe("even when one handler is added inside another handler", function () {
                testFulfilled(dummy, function (promise, done) {
                    var handler1 = sinon.spy(function handler1() {});
                    var handler2 = sinon.spy(function handler2() {});
                    var handler3 = sinon.spy(function handler3() {});

                    promise.then(function () {
                        handler1();
                        promise.then(handler3);
                    });
                    promise.then(handler2);

                    promise.then(function () {
                        // Give implementations a bit of extra time to flush their internal queue, if necessary.
                        setTimeout(function () {
                            sinon.assert.callOrder(handler1, handler2, handler3);
                            done();
                        }, 15);
                    });
                });
            });
        });
    });

    describe("2.2.6.2: If/when `promise` is rejected, all respective `onRejected` callbacks must execute in the " +
             "order of their originating calls to `then`.", function () {
        describe("multiple boring rejection handlers", function () {
            testRejected(sentinel, function (promise, done) {
                var handler1 = sinon.stub().returns(other);
                var handler2 = sinon.stub().returns(other);
                var handler3 = sinon.stub().returns(other);

                var spy = sinon.spy();
                promise.then(spy, handler1);
                promise.then(spy, handler2);
                promise.then(spy, handler3);

                promise.then(null, function (reason) {
                    assert.strictEqual(reason, sentinel);

                    sinon.assert.calledWith(handler1, sinon.match.same(sentinel));
                    sinon.assert.calledWith(handler2, sinon.match.same(sentinel));
                    sinon.assert.calledWith(handler3, sinon.match.same(sentinel));
                    sinon.assert.notCalled(spy);

                    done();
                });
            });
        });

        describe("multiple rejection handlers, one of which throws", function () {
            testRejected(sentinel, function (promise, done) {
                var handler1 = sinon.stub().returns(other);
                var handler2 = sinon.stub().throws(other);
                var handler3 = sinon.stub().returns(other);

                var spy = sinon.spy();
                promise.then(spy, handler1);
                promise.then(spy, handler2);
                promise.then(spy, handler3);

                promise.then(null, function (reason) {
                    assert.strictEqual(reason, sentinel);

                    sinon.assert.calledWith(handler1, sinon.match.same(sentinel));
                    sinon.assert.calledWith(handler2, sinon.match.same(sentinel));
                    sinon.assert.calledWith(handler3, sinon.match.same(sentinel));
                    sinon.assert.notCalled(spy);

                    done();
                });
            });
        });

        describe("results in multiple branching chains with their own fulfillment values", function () {
            testRejected(sentinel, function (promise, done) {
                var semiDone = callbackAggregator(3, done);

                promise.then(null, function () {
                    return sentinel;
                }).then(function (value) {
                    assert.strictEqual(value, sentinel);
                    semiDone();
                });

                promise.then(null, function () {
                    throw sentinel2;
                }).then(null, function (reason) {
                    assert.strictEqual(reason, sentinel2);
                    semiDone();
                });

                promise.then(null, function () {
                    return sentinel3;
                }).then(function (value) {
                    assert.strictEqual(value, sentinel3);
                    semiDone();
                });
            });
        });

        describe("`onRejected` handlers are called in the original order", function () {
            testRejected(dummy, function (promise, done) {
                var handler1 = sinon.spy(function handler1() {});
                var handler2 = sinon.spy(function handler2() {});
                var handler3 = sinon.spy(function handler3() {});

                promise.then(null, handler1);
                promise.then(null, handler2);
                promise.then(null, handler3);

                promise.then(null, function () {
                    sinon.assert.callOrder(handler1, handler2, handler3);
                    done();
                });
            });

            describe("even when one handler is added inside another handler", function () {
                testRejected(dummy, function (promise, done) {
                    var handler1 = sinon.spy(function handler1() {});
                    var handler2 = sinon.spy(function handler2() {});
                    var handler3 = sinon.spy(function handler3() {});

                    promise.then(null, function () {
                        handler1();
                        promise.then(null, handler3);
                    });
                    promise.then(null, handler2);

                    promise.then(null, function () {
                        // Give implementations a bit of extra time to flush their internal queue, if necessary.
                        setTimeout(function () {
                            sinon.assert.callOrder(handler1, handler2, handler3);
                            done();
                        }, 15);
                    });
                });
            });
        });
    });
});

},{"./helpers/testThreeCases":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/testThreeCases.js","assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js","sinon":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.2.7.js":[function(require,module,exports){
(function (global){
"use strict";

var assert = require("assert");
var testFulfilled = require("./helpers/testThreeCases").testFulfilled;
var testRejected = require("./helpers/testThreeCases").testRejected;
var reasons = require("./helpers/reasons");

var adapter = global.adapter;
var deferred = adapter.deferred;

var dummy = { dummy: "dummy" }; // we fulfill or reject with this when we don't intend to test against it
var sentinel = { sentinel: "sentinel" }; // a sentinel fulfillment value to test for with strict equality
var other = { other: "other" }; // a value we don't want to be strict equal to

describe("2.2.7: `then` must return a promise: `promise2 = promise1.then(onFulfilled, onRejected)`", function () {
    specify("is a promise", function () {
        var promise1 = deferred().promise;
        var promise2 = promise1.then();

        assert(typeof promise2 === "object" || typeof promise2 === "function");
        assert.notStrictEqual(promise2, null);
        assert.strictEqual(typeof promise2.then, "function");
    });

    describe("2.2.7.1: If either `onFulfilled` or `onRejected` returns a value `x`, run the Promise Resolution " +
             "Procedure `[[Resolve]](promise2, x)`", function () {
        specify("see separate 3.3 tests", function () { });
    });

    describe("2.2.7.2: If either `onFulfilled` or `onRejected` throws an exception `e`, `promise2` must be rejected " +
             "with `e` as the reason.", function () {
        function testReason(expectedReason, stringRepresentation) {
            describe("The reason is " + stringRepresentation, function () {
                testFulfilled(dummy, function (promise1, done) {
                    var promise2 = promise1.then(function onFulfilled() {
                        throw expectedReason;
                    });

                    promise2.then(null, function onPromise2Rejected(actualReason) {
                        assert.strictEqual(actualReason, expectedReason);
                        done();
                    });
                });
                testRejected(dummy, function (promise1, done) {
                    var promise2 = promise1.then(null, function onRejected() {
                        throw expectedReason;
                    });

                    promise2.then(null, function onPromise2Rejected(actualReason) {
                        assert.strictEqual(actualReason, expectedReason);
                        done();
                    });
                });
            });
        }

        Object.keys(reasons).forEach(function (stringRepresentation) {
            testReason(reasons[stringRepresentation], stringRepresentation);
        });
    });

    describe("2.2.7.3: If `onFulfilled` is not a function and `promise1` is fulfilled, `promise2` must be fulfilled " +
             "with the same value.", function () {

        function testNonFunction(nonFunction, stringRepresentation) {
            describe("`onFulfilled` is " + stringRepresentation, function () {
                testFulfilled(sentinel, function (promise1, done) {
                    var promise2 = promise1.then(nonFunction);

                    promise2.then(function onPromise2Fulfilled(value) {
                        assert.strictEqual(value, sentinel);
                        done();
                    });
                });
            });
        }

        testNonFunction(undefined, "`undefined`");
        testNonFunction(null, "`null`");
        testNonFunction(false, "`false`");
        testNonFunction(5, "`5`");
        testNonFunction({}, "an object");
        testNonFunction([function () { return other; }], "an array containing a function");
    });

    describe("2.2.7.4: If `onRejected` is not a function and `promise1` is rejected, `promise2` must be rejected " +
             "with the same reason.", function () {

        function testNonFunction(nonFunction, stringRepresentation) {
            describe("`onRejected` is " + stringRepresentation, function () {
                testRejected(sentinel, function (promise1, done) {
                    var promise2 = promise1.then(null, nonFunction);

                    promise2.then(null, function onPromise2Rejected(reason) {
                        assert.strictEqual(reason, sentinel);
                        done();
                    });
                });
            });
        }

        testNonFunction(undefined, "`undefined`");
        testNonFunction(null, "`null`");
        testNonFunction(false, "`false`");
        testNonFunction(5, "`5`");
        testNonFunction({}, "an object");
        testNonFunction([function () { return other; }], "an array containing a function");
    });
});

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./helpers/reasons":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/reasons.js","./helpers/testThreeCases":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/testThreeCases.js","assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.3.1.js":[function(require,module,exports){
(function (global){
"use strict";

var assert = require("assert");

var adapter = global.adapter;
var resolved = adapter.resolved;
var rejected = adapter.rejected;

var dummy = { dummy: "dummy" }; // we fulfill or reject with this when we don't intend to test against it

describe("2.3.1: If `promise` and `x` refer to the same object, reject `promise` with a `TypeError' as the reason.",
         function () {
    specify("via return from a fulfilled promise", function (done) {
        var promise = resolved(dummy).then(function () {
            return promise;
        });

        promise.then(null, function (reason) {
            assert(reason instanceof TypeError);
            done();
        });
    });

    specify("via return from a rejected promise", function (done) {
        var promise = rejected(dummy).then(null, function () {
            return promise;
        });

        promise.then(null, function (reason) {
            assert(reason instanceof TypeError);
            done();
        });
    });
});

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.3.2.js":[function(require,module,exports){
(function (global){
"use strict";

var assert = require("assert");

var adapter = global.adapter;
var resolved = adapter.resolved;
var rejected = adapter.rejected;
var deferred = adapter.deferred;

var dummy = { dummy: "dummy" }; // we fulfill or reject with this when we don't intend to test against it
var sentinel = { sentinel: "sentinel" }; // a sentinel fulfillment value to test for with strict equality

function testPromiseResolution(xFactory, test) {
    specify("via return from a fulfilled promise", function (done) {
        var promise = resolved(dummy).then(function onBasePromiseFulfilled() {
            return xFactory();
        });

        test(promise, done);
    });

    specify("via return from a rejected promise", function (done) {
        var promise = rejected(dummy).then(null, function onBasePromiseRejected() {
            return xFactory();
        });

        test(promise, done);
    });
}

describe("2.3.2: If `x` is a promise, adopt its state", function () {
    describe("2.3.2.1: If `x` is pending, `promise` must remain pending until `x` is fulfilled or rejected.",
             function () {
        function xFactory() {
            return deferred().promise;
        }

        testPromiseResolution(xFactory, function (promise, done) {
            var wasFulfilled = false;
            var wasRejected = false;

            promise.then(
                function onPromiseFulfilled() {
                    wasFulfilled = true;
                },
                function onPromiseRejected() {
                    wasRejected = true;
                }
            );

            setTimeout(function () {
                assert.strictEqual(wasFulfilled, false);
                assert.strictEqual(wasRejected, false);
                done();
            }, 100);
        });
    });

    describe("2.3.2.2: If/when `x` is fulfilled, fulfill `promise` with the same value.", function () {
        describe("`x` is already-fulfilled", function () {
            function xFactory() {
                return resolved(sentinel);
            }

            testPromiseResolution(xFactory, function (promise, done) {
                promise.then(function onPromiseFulfilled(value) {
                    assert.strictEqual(value, sentinel);
                    done();
                });
            });
        });

        describe("`x` is eventually-fulfilled", function () {
            var d = null;

            function xFactory() {
                d = deferred();
                setTimeout(function () {
                    d.resolve(sentinel);
                }, 50);
                return d.promise;
            }

            testPromiseResolution(xFactory, function (promise, done) {
                promise.then(function onPromiseFulfilled(value) {
                    assert.strictEqual(value, sentinel);
                    done();
                });
            });
        });
    });

    describe("2.3.2.3: If/when `x` is rejected, reject `promise` with the same reason.", function () {
        describe("`x` is already-rejected", function () {
            function xFactory() {
                return rejected(sentinel);
            }

            testPromiseResolution(xFactory, function (promise, done) {
                promise.then(null, function onPromiseRejected(reason) {
                    assert.strictEqual(reason, sentinel);
                    done();
                });
            });
        });

        describe("`x` is eventually-rejected", function () {
            var d = null;

            function xFactory() {
                d = deferred();
                setTimeout(function () {
                    d.reject(sentinel);
                }, 50);
                return d.promise;
            }

            testPromiseResolution(xFactory, function (promise, done) {
                promise.then(null, function onPromiseRejected(reason) {
                    assert.strictEqual(reason, sentinel);
                    done();
                });
            });
        });
    });
});

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.3.3.js":[function(require,module,exports){
(function (global){
"use strict";

var assert = require("assert");
var thenables = require("./helpers/thenables");
var reasons = require("./helpers/reasons");

var adapter = global.adapter;
var resolved = adapter.resolved;
var rejected = adapter.rejected;
var deferred = adapter.deferred;

var dummy = { dummy: "dummy" }; // we fulfill or reject with this when we don't intend to test against it
var sentinel = { sentinel: "sentinel" }; // a sentinel fulfillment value to test for with strict equality
var other = { other: "other" }; // a value we don't want to be strict equal to
var sentinelArray = [sentinel]; // a sentinel fulfillment value to test when we need an array

function testPromiseResolution(xFactory, test) {
    specify("via return from a fulfilled promise", function (done) {
        var promise = resolved(dummy).then(function onBasePromiseFulfilled() {
            return xFactory();
        });

        test(promise, done);
    });

    specify("via return from a rejected promise", function (done) {
        var promise = rejected(dummy).then(null, function onBasePromiseRejected() {
            return xFactory();
        });

        test(promise, done);
    });
}

function testCallingResolvePromise(yFactory, stringRepresentation, test) {
    describe("`y` is " + stringRepresentation, function () {
        describe("`then` calls `resolvePromise` synchronously", function () {
            function xFactory() {
                return {
                    then: function (resolvePromise) {
                        resolvePromise(yFactory());
                    }
                };
            }

            testPromiseResolution(xFactory, test);
        });

        describe("`then` calls `resolvePromise` asynchronously", function () {
            function xFactory() {
                return {
                    then: function (resolvePromise) {
                        setTimeout(function () {
                            resolvePromise(yFactory());
                        }, 0);
                    }
                };
            }

            testPromiseResolution(xFactory, test);
        });
    });
}

function testCallingRejectPromise(r, stringRepresentation, test) {
    describe("`r` is " + stringRepresentation, function () {
        describe("`then` calls `rejectPromise` synchronously", function () {
            function xFactory() {
                return {
                    then: function (resolvePromise, rejectPromise) {
                        rejectPromise(r);
                    }
                };
            }

            testPromiseResolution(xFactory, test);
        });

        describe("`then` calls `rejectPromise` asynchronously", function () {
            function xFactory() {
                return {
                    then: function (resolvePromise, rejectPromise) {
                        setTimeout(function () {
                            rejectPromise(r);
                        }, 0);
                    }
                };
            }

            testPromiseResolution(xFactory, test);
        });
    });
}

function testCallingResolvePromiseFulfillsWith(yFactory, stringRepresentation, fulfillmentValue) {
    testCallingResolvePromise(yFactory, stringRepresentation, function (promise, done) {
        promise.then(function onPromiseFulfilled(value) {
            assert.strictEqual(value, fulfillmentValue);
            done();
        });
    });
}

function testCallingResolvePromiseRejectsWith(yFactory, stringRepresentation, rejectionReason) {
    testCallingResolvePromise(yFactory, stringRepresentation, function (promise, done) {
        promise.then(null, function onPromiseRejected(reason) {
            assert.strictEqual(reason, rejectionReason);
            done();
        });
    });
}

function testCallingRejectPromiseRejectsWith(reason, stringRepresentation) {
    testCallingRejectPromise(reason, stringRepresentation, function (promise, done) {
        promise.then(null, function onPromiseRejected(rejectionReason) {
            assert.strictEqual(rejectionReason, reason);
            done();
        });
    });
}

describe("2.3.3: Otherwise, if `x` is an object or function,", function () {
    describe("2.3.3.1: Let `then` be `x.then`", function () {
        describe("`x` is an object with null prototype", function () {
            var numberOfTimesThenWasRetrieved = null;

            beforeEach(function () {
                numberOfTimesThenWasRetrieved = 0;
            });

            function xFactory() {
                return Object.create(null, {
                    then: {
                        get: function () {
                            ++numberOfTimesThenWasRetrieved;
                            return function thenMethodForX(onFulfilled) {
                                onFulfilled();
                            };
                        }
                    }
                });
            }

            testPromiseResolution(xFactory, function (promise, done) {
                promise.then(function () {
                    assert.strictEqual(numberOfTimesThenWasRetrieved, 1);
                    done();
                });
            });
        });

        describe("`x` is an object with normal Object.prototype", function () {
            var numberOfTimesThenWasRetrieved = null;

            beforeEach(function () {
                numberOfTimesThenWasRetrieved = 0;
            });

            function xFactory() {
                return Object.create(Object.prototype, {
                    then: {
                        get: function () {
                            ++numberOfTimesThenWasRetrieved;
                            return function thenMethodForX(onFulfilled) {
                                onFulfilled();
                            };
                        }
                    }
                });
            }

            testPromiseResolution(xFactory, function (promise, done) {
                promise.then(function () {
                    assert.strictEqual(numberOfTimesThenWasRetrieved, 1);
                    done();
                });
            });
        });

        describe("`x` is a function", function () {
            var numberOfTimesThenWasRetrieved = null;

            beforeEach(function () {
                numberOfTimesThenWasRetrieved = 0;
            });

            function xFactory() {
                function x() { }

                Object.defineProperty(x, "then", {
                    get: function () {
                        ++numberOfTimesThenWasRetrieved;
                        return function thenMethodForX(onFulfilled) {
                            onFulfilled();
                        };
                    }
                });

                return x;
            }

            testPromiseResolution(xFactory, function (promise, done) {
                promise.then(function () {
                    assert.strictEqual(numberOfTimesThenWasRetrieved, 1);
                    done();
                });
            });
        });
    });

    describe("2.3.3.2: If retrieving the property `x.then` results in a thrown exception `e`, reject `promise` with " +
             "`e` as the reason.", function () {
        function testRejectionViaThrowingGetter(e, stringRepresentation) {
            function xFactory() {
                return Object.create(Object.prototype, {
                    then: {
                        get: function () {
                            throw e;
                        }
                    }
                });
            }

            describe("`e` is " + stringRepresentation, function () {
                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(null, function (reason) {
                        assert.strictEqual(reason, e);
                        done();
                    });
                });
            });
        }

        Object.keys(reasons).forEach(function (stringRepresentation) {
            testRejectionViaThrowingGetter(reasons[stringRepresentation], stringRepresentation);
        });
    });

    describe("2.3.3.3: If `then` is a function, call it with `x` as `this`, first argument `resolvePromise`, and " +
             "second argument `rejectPromise`", function () {
        describe("Calls with `x` as `this` and two function arguments", function () {
            function xFactory() {
                var x = {
                    then: function (onFulfilled, onRejected) {
                        assert.strictEqual(this, x);
                        assert.strictEqual(typeof onFulfilled, "function");
                        assert.strictEqual(typeof onRejected, "function");
                        onFulfilled();
                    }
                };
                return x;
            }

            testPromiseResolution(xFactory, function (promise, done) {
                promise.then(function () {
                    done();
                });
            });
        });

        describe("Uses the original value of `then`", function () {
            var numberOfTimesThenWasRetrieved = null;

            beforeEach(function () {
                numberOfTimesThenWasRetrieved = 0;
            });

            function xFactory() {
                return Object.create(Object.prototype, {
                    then: {
                        get: function () {
                            if (numberOfTimesThenWasRetrieved === 0) {
                                return function (onFulfilled) {
                                    onFulfilled();
                                };
                            }
                            return null;
                        }
                    }
                });
            }

            testPromiseResolution(xFactory, function (promise, done) {
                promise.then(function () {
                    done();
                });
            });
        });

        describe("2.3.3.3.1: If/when `resolvePromise` is called with value `y`, run `[[Resolve]](promise, y)`",
                 function () {
            describe("`y` is not a thenable", function () {
                testCallingResolvePromiseFulfillsWith(function () { return undefined; }, "`undefined`", undefined);
                testCallingResolvePromiseFulfillsWith(function () { return null; }, "`null`", null);
                testCallingResolvePromiseFulfillsWith(function () { return false; }, "`false`", false);
                testCallingResolvePromiseFulfillsWith(function () { return 5; }, "`5`", 5);
                testCallingResolvePromiseFulfillsWith(function () { return sentinel; }, "an object", sentinel);
                testCallingResolvePromiseFulfillsWith(function () { return sentinelArray; }, "an array", sentinelArray);
            });

            describe("`y` is a thenable", function () {
                Object.keys(thenables.fulfilled).forEach(function (stringRepresentation) {
                    function yFactory() {
                        return thenables.fulfilled[stringRepresentation](sentinel);
                    }

                    testCallingResolvePromiseFulfillsWith(yFactory, stringRepresentation, sentinel);
                });

                Object.keys(thenables.rejected).forEach(function (stringRepresentation) {
                    function yFactory() {
                        return thenables.rejected[stringRepresentation](sentinel);
                    }

                    testCallingResolvePromiseRejectsWith(yFactory, stringRepresentation, sentinel);
                });
            });

            describe("`y` is a thenable for a thenable", function () {
                Object.keys(thenables.fulfilled).forEach(function (outerStringRepresentation) {
                    var outerThenableFactory = thenables.fulfilled[outerStringRepresentation];

                    Object.keys(thenables.fulfilled).forEach(function (innerStringRepresentation) {
                        var innerThenableFactory = thenables.fulfilled[innerStringRepresentation];

                        var stringRepresentation = outerStringRepresentation + " for " + innerStringRepresentation;

                        function yFactory() {
                            return outerThenableFactory(innerThenableFactory(sentinel));
                        }

                        testCallingResolvePromiseFulfillsWith(yFactory, stringRepresentation, sentinel);
                    });

                    Object.keys(thenables.rejected).forEach(function (innerStringRepresentation) {
                        var innerThenableFactory = thenables.rejected[innerStringRepresentation];

                        var stringRepresentation = outerStringRepresentation + " for " + innerStringRepresentation;

                        function yFactory() {
                            return outerThenableFactory(innerThenableFactory(sentinel));
                        }

                        testCallingResolvePromiseRejectsWith(yFactory, stringRepresentation, sentinel);
                    });
                });
            });
        });

        describe("2.3.3.3.2: If/when `rejectPromise` is called with reason `r`, reject `promise` with `r`",
                 function () {
            Object.keys(reasons).forEach(function (stringRepresentation) {
                testCallingRejectPromiseRejectsWith(reasons[stringRepresentation], stringRepresentation);
            });
        });

        describe("2.3.3.3.3: If both `resolvePromise` and `rejectPromise` are called, or multiple calls to the same " +
                 "argument are made, the first call takes precedence, and any further calls are ignored.",
                 function () {
            describe("calling `resolvePromise` then `rejectPromise`, both synchronously", function () {
                function xFactory() {
                    return {
                        then: function (resolvePromise, rejectPromise) {
                            resolvePromise(sentinel);
                            rejectPromise(other);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(function (value) {
                        assert.strictEqual(value, sentinel);
                        done();
                    });
                });
            });

            describe("calling `resolvePromise` synchronously then `rejectPromise` asynchronously", function () {
                function xFactory() {
                    return {
                        then: function (resolvePromise, rejectPromise) {
                            resolvePromise(sentinel);

                            setTimeout(function () {
                                rejectPromise(other);
                            }, 0);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(function (value) {
                        assert.strictEqual(value, sentinel);
                        done();
                    });
                });
            });

            describe("calling `resolvePromise` then `rejectPromise`, both asynchronously", function () {
                function xFactory() {
                    return {
                        then: function (resolvePromise, rejectPromise) {
                            setTimeout(function () {
                                resolvePromise(sentinel);
                            }, 0);

                            setTimeout(function () {
                                rejectPromise(other);
                            }, 0);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(function (value) {
                        assert.strictEqual(value, sentinel);
                        done();
                    });
                });
            });

            describe("calling `resolvePromise` with an asynchronously-fulfilled promise, then calling " +
                     "`rejectPromise`, both synchronously", function () {
                function xFactory() {
                    var d = deferred();
                    setTimeout(function () {
                        d.resolve(sentinel);
                    }, 50);

                    return {
                        then: function (resolvePromise, rejectPromise) {
                            resolvePromise(d.promise);
                            rejectPromise(other);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(function (value) {
                        assert.strictEqual(value, sentinel);
                        done();
                    });
                });
            });

            describe("calling `resolvePromise` with an asynchronously-rejected promise, then calling " +
                     "`rejectPromise`, both synchronously", function () {
                function xFactory() {
                    var d = deferred();
                    setTimeout(function () {
                        d.reject(sentinel);
                    }, 50);

                    return {
                        then: function (resolvePromise, rejectPromise) {
                            resolvePromise(d.promise);
                            rejectPromise(other);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(null, function (reason) {
                        assert.strictEqual(reason, sentinel);
                        done();
                    });
                });
            });

            describe("calling `rejectPromise` then `resolvePromise`, both synchronously", function () {
                function xFactory() {
                    return {
                        then: function (resolvePromise, rejectPromise) {
                            rejectPromise(sentinel);
                            resolvePromise(other);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(null, function (reason) {
                        assert.strictEqual(reason, sentinel);
                        done();
                    });
                });
            });

            describe("calling `rejectPromise` synchronously then `resolvePromise` asynchronously", function () {
                function xFactory() {
                    return {
                        then: function (resolvePromise, rejectPromise) {
                            rejectPromise(sentinel);

                            setTimeout(function () {
                                resolvePromise(other);
                            }, 0);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(null, function (reason) {
                        assert.strictEqual(reason, sentinel);
                        done();
                    });
                });
            });

            describe("calling `rejectPromise` then `resolvePromise`, both asynchronously", function () {
                function xFactory() {
                    return {
                        then: function (resolvePromise, rejectPromise) {
                            setTimeout(function () {
                                rejectPromise(sentinel);
                            }, 0);

                            setTimeout(function () {
                                resolvePromise(other);
                            }, 0);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(null, function (reason) {
                        assert.strictEqual(reason, sentinel);
                        done();
                    });
                });
            });

            describe("calling `resolvePromise` twice synchronously", function () {
                function xFactory() {
                    return {
                        then: function (resolvePromise) {
                            resolvePromise(sentinel);
                            resolvePromise(other);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(function (value) {
                        assert.strictEqual(value, sentinel);
                        done();
                    });
                });
            });

            describe("calling `resolvePromise` twice, first synchronously then asynchronously", function () {
                function xFactory() {
                    return {
                        then: function (resolvePromise) {
                            resolvePromise(sentinel);

                            setTimeout(function () {
                                resolvePromise(other);
                            }, 0);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(function (value) {
                        assert.strictEqual(value, sentinel);
                        done();
                    });
                });
            });

            describe("calling `resolvePromise` twice, both times asynchronously", function () {
                function xFactory() {
                    return {
                        then: function (resolvePromise) {
                            setTimeout(function () {
                                resolvePromise(sentinel);
                            }, 0);

                            setTimeout(function () {
                                resolvePromise(other);
                            }, 0);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(function (value) {
                        assert.strictEqual(value, sentinel);
                        done();
                    });
                });
            });

            describe("calling `resolvePromise` with an asynchronously-fulfilled promise, then calling it again, both " +
                     "times synchronously", function () {
                function xFactory() {
                    var d = deferred();
                    setTimeout(function () {
                        d.resolve(sentinel);
                    }, 50);

                    return {
                        then: function (resolvePromise) {
                            resolvePromise(d.promise);
                            resolvePromise(other);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(function (value) {
                        assert.strictEqual(value, sentinel);
                        done();
                    });
                });
            });

            describe("calling `resolvePromise` with an asynchronously-rejected promise, then calling it again, both " +
                     "times synchronously", function () {
                function xFactory() {
                    var d = deferred();
                    setTimeout(function () {
                        d.reject(sentinel);
                    }, 50);

                    return {
                        then: function (resolvePromise) {
                            resolvePromise(d.promise);
                            resolvePromise(other);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(null, function (reason) {
                        assert.strictEqual(reason, sentinel);
                        done();
                    });
                });
            });

            describe("calling `rejectPromise` twice synchronously", function () {
                function xFactory() {
                    return {
                        then: function (resolvePromise, rejectPromise) {
                            rejectPromise(sentinel);
                            rejectPromise(other);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(null, function (reason) {
                        assert.strictEqual(reason, sentinel);
                        done();
                    });
                });
            });

            describe("calling `rejectPromise` twice, first synchronously then asynchronously", function () {
                function xFactory() {
                    return {
                        then: function (resolvePromise, rejectPromise) {
                            rejectPromise(sentinel);

                            setTimeout(function () {
                                rejectPromise(other);
                            }, 0);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(null, function (reason) {
                        assert.strictEqual(reason, sentinel);
                        done();
                    });
                });
            });

            describe("calling `rejectPromise` twice, both times asynchronously", function () {
                function xFactory() {
                    return {
                        then: function (resolvePromise, rejectPromise) {
                            setTimeout(function () {
                                rejectPromise(sentinel);
                            }, 0);

                            setTimeout(function () {
                                rejectPromise(other);
                            }, 0);
                        }
                    };
                }

                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(null, function (reason) {
                        assert.strictEqual(reason, sentinel);
                        done();
                    });
                });
            });

            describe("saving and abusing `resolvePromise` and `rejectPromise`", function () {
                var savedResolvePromise, savedRejectPromise;

                function xFactory() {
                    return {
                        then: function (resolvePromise, rejectPromise) {
                            savedResolvePromise = resolvePromise;
                            savedRejectPromise = rejectPromise;
                        }
                    };
                }

                beforeEach(function () {
                    savedResolvePromise = null;
                    savedRejectPromise = null;
                });

                testPromiseResolution(xFactory, function (promise, done) {
                    var timesFulfilled = 0;
                    var timesRejected = 0;

                    promise.then(
                        function () {
                            ++timesFulfilled;
                        },
                        function () {
                            ++timesRejected;
                        }
                    );

                    if (savedResolvePromise && savedRejectPromise) {
                        savedResolvePromise(dummy);
                        savedResolvePromise(dummy);
                        savedRejectPromise(dummy);
                        savedRejectPromise(dummy);
                    }

                    setTimeout(function () {
                        savedResolvePromise(dummy);
                        savedResolvePromise(dummy);
                        savedRejectPromise(dummy);
                        savedRejectPromise(dummy);
                    }, 50);

                    setTimeout(function () {
                        assert.strictEqual(timesFulfilled, 1);
                        assert.strictEqual(timesRejected, 0);
                        done();
                    }, 100);
                });
            });
        });

        describe("2.3.3.3.4: If calling `then` throws an exception `e`,", function () {
            describe("2.3.3.3.4.1: If `resolvePromise` or `rejectPromise` have been called, ignore it.", function () {
                describe("`resolvePromise` was called with a non-thenable", function () {
                    function xFactory() {
                        return {
                            then: function (resolvePromise) {
                                resolvePromise(sentinel);
                                throw other;
                            }
                        };
                    }

                    testPromiseResolution(xFactory, function (promise, done) {
                        promise.then(function (value) {
                            assert.strictEqual(value, sentinel);
                            done();
                        });
                    });
                });

                describe("`resolvePromise` was called with an asynchronously-fulfilled promise", function () {
                    function xFactory() {
                        var d = deferred();
                        setTimeout(function () {
                            d.resolve(sentinel);
                        }, 50);

                        return {
                            then: function (resolvePromise) {
                                resolvePromise(d.promise);
                                throw other;
                            }
                        };
                    }

                    testPromiseResolution(xFactory, function (promise, done) {
                        promise.then(function (value) {
                            assert.strictEqual(value, sentinel);
                            done();
                        });
                    });
                });

                describe("`resolvePromise` was called with an asynchronously-rejected promise", function () {
                    function xFactory() {
                        var d = deferred();
                        setTimeout(function () {
                            d.reject(sentinel);
                        }, 50);

                        return {
                            then: function (resolvePromise) {
                                resolvePromise(d.promise);
                                throw other;
                            }
                        };
                    }

                    testPromiseResolution(xFactory, function (promise, done) {
                        promise.then(null, function (reason) {
                            assert.strictEqual(reason, sentinel);
                            done();
                        });
                    });
                });

                describe("`rejectPromise` was called", function () {
                    function xFactory() {
                        return {
                            then: function (resolvePromise, rejectPromise) {
                                rejectPromise(sentinel);
                                throw other;
                            }
                        };
                    }

                    testPromiseResolution(xFactory, function (promise, done) {
                        promise.then(null, function (reason) {
                            assert.strictEqual(reason, sentinel);
                            done();
                        });
                    });
                });

                describe("`resolvePromise` then `rejectPromise` were called", function () {
                    function xFactory() {
                        return {
                            then: function (resolvePromise, rejectPromise) {
                                resolvePromise(sentinel);
                                rejectPromise(other);
                                throw other;
                            }
                        };
                    }

                    testPromiseResolution(xFactory, function (promise, done) {
                        promise.then(function (value) {
                            assert.strictEqual(value, sentinel);
                            done();
                        });
                    });
                });

                describe("`rejectPromise` then `resolvePromise` were called", function () {
                    function xFactory() {
                        return {
                            then: function (resolvePromise, rejectPromise) {
                                rejectPromise(sentinel);
                                resolvePromise(other);
                                throw other;
                            }
                        };
                    }

                    testPromiseResolution(xFactory, function (promise, done) {
                        promise.then(null, function (reason) {
                            assert.strictEqual(reason, sentinel);
                            done();
                        });
                    });
                });
            });

            describe("2.3.3.3.4.2: Otherwise, reject `promise` with `e` as the reason.", function () {
                describe("straightforward case", function () {
                    function xFactory() {
                        return {
                            then: function () {
                                throw sentinel;
                            }
                        };
                    }

                    testPromiseResolution(xFactory, function (promise, done) {
                        promise.then(null, function (reason) {
                            assert.strictEqual(reason, sentinel);
                            done();
                        });
                    });
                });

                describe("`resolvePromise` is called asynchronously before the `throw`", function () {
                    function xFactory() {
                        return {
                            then: function (resolvePromise) {
                                setTimeout(function () {
                                    resolvePromise(other);
                                }, 0);
                                throw sentinel;
                            }
                        };
                    }

                    testPromiseResolution(xFactory, function (promise, done) {
                        promise.then(null, function (reason) {
                            assert.strictEqual(reason, sentinel);
                            done();
                        });
                    });
                });

                describe("`rejectPromise` is called asynchronously before the `throw`", function () {
                    function xFactory() {
                        return {
                            then: function (resolvePromise, rejectPromise) {
                                setTimeout(function () {
                                    rejectPromise(other);
                                }, 0);
                                throw sentinel;
                            }
                        };
                    }

                    testPromiseResolution(xFactory, function (promise, done) {
                        promise.then(null, function (reason) {
                            assert.strictEqual(reason, sentinel);
                            done();
                        });
                    });
                });
            });
        });
    });

    describe("2.3.3.4: If `then` is not a function, fulfill promise with `x`", function () {
        function testFulfillViaNonFunction(then, stringRepresentation) {
            var x = null;

            beforeEach(function () {
                x = { then: then };
            });

            function xFactory() {
                return x;
            }

            describe("`then` is " + stringRepresentation, function () {
                testPromiseResolution(xFactory, function (promise, done) {
                    promise.then(function (value) {
                        assert.strictEqual(value, x);
                        done();
                    });
                });
            });
        }

        testFulfillViaNonFunction(5, "`5`");
        testFulfillViaNonFunction({}, "an object");
        testFulfillViaNonFunction([function () { }], "an array containing a function");
        testFulfillViaNonFunction(/a-b/i, "a regular expression");
        testFulfillViaNonFunction(Object.create(Function.prototype), "an object inheriting from `Function.prototype`");
    });
});

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./helpers/reasons":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/reasons.js","./helpers/thenables":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/thenables.js","assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/2.3.4.js":[function(require,module,exports){
"use strict";

var assert = require("assert");
var testFulfilled = require("./helpers/testThreeCases").testFulfilled;
var testRejected = require("./helpers/testThreeCases").testRejected;

var dummy = { dummy: "dummy" }; // we fulfill or reject with this when we don't intend to test against it

describe("2.3.4: If `x` is not an object or function, fulfill `promise` with `x`", function () {
    function testValue(expectedValue, stringRepresentation, beforeEachHook, afterEachHook) {
        describe("The value is " + stringRepresentation, function () {
            if (typeof beforeEachHook === "function") {
                beforeEach(beforeEachHook);
            }
            if (typeof afterEachHook === "function") {
                afterEach(afterEachHook);
            }

            testFulfilled(dummy, function (promise1, done) {
                var promise2 = promise1.then(function onFulfilled() {
                    return expectedValue;
                });

                promise2.then(function onPromise2Fulfilled(actualValue) {
                    assert.strictEqual(actualValue, expectedValue);
                    done();
                });
            });
            testRejected(dummy, function (promise1, done) {
                var promise2 = promise1.then(null, function onRejected() {
                    return expectedValue;
                });

                promise2.then(function onPromise2Fulfilled(actualValue) {
                    assert.strictEqual(actualValue, expectedValue);
                    done();
                });
            });
        });
    }

    testValue(undefined, "`undefined`");
    testValue(null, "`null`");
    testValue(false, "`false`");
    testValue(true, "`true`");
    testValue(0, "`0`");

    testValue(
        true,
        "`true` with `Boolean.prototype` modified to have a `then` method",
        function () {
            Boolean.prototype.then = function () {};
        },
        function () {
            delete Boolean.prototype.then;
        }
    );

    testValue(
        1,
        "`1` with `Number.prototype` modified to have a `then` method",
        function () {
            Number.prototype.then = function () {};
        },
        function () {
            delete Number.prototype.then;
        }
    );
});

},{"./helpers/testThreeCases":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/testThreeCases.js","assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/reasons.js":[function(require,module,exports){
(function (global){
"use strict";

// This module exports some valid rejection reason factories, keyed by human-readable versions of their names.

var adapter = global.adapter;
var resolved = adapter.resolved;
var rejected = adapter.rejected;

var dummy = { dummy: "dummy" };

exports["`undefined`"] = function () {
    return undefined;
};

exports["`null`"] = function () {
    return null;
};

exports["`false`"] = function () {
    return false;
};

exports["`0`"] = function () {
    return 0;
};

exports["an error"] = function () {
    return new Error();
};

exports["an error without a stack"] = function () {
    var error = new Error();
    delete error.stack;

    return error;
};

exports["a date"] = function () {
    return new Date();
};

exports["an object"] = function () {
    return {};
};

exports["an always-pending thenable"] = function () {
    return { then: function () { } };
};

exports["a fulfilled promise"] = function () {
    return resolved(dummy);
};

exports["a rejected promise"] = function () {
    return rejected(dummy);
};

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/testThreeCases.js":[function(require,module,exports){
(function (global){
"use strict";

var adapter = global.adapter;
var resolved = adapter.resolved;
var rejected = adapter.rejected;
var deferred = adapter.deferred;

exports.testFulfilled = function (value, test) {
    specify("already-fulfilled", function (done) {
        test(resolved(value), done);
    });

    specify("immediately-fulfilled", function (done) {
        var d = deferred();
        test(d.promise, done);
        d.resolve(value);
    });

    specify("eventually-fulfilled", function (done) {
        var d = deferred();
        test(d.promise, done);
        setTimeout(function () {
            d.resolve(value);
        }, 50);
    });
};

exports.testRejected = function (reason, test) {
    specify("already-rejected", function (done) {
        test(rejected(reason), done);
    });

    specify("immediately-rejected", function (done) {
        var d = deferred();
        test(d.promise, done);
        d.reject(reason);
    });

    specify("eventually-rejected", function (done) {
        var d = deferred();
        test(d.promise, done);
        setTimeout(function () {
            d.reject(reason);
        }, 50);
    });
};

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/lib/tests/helpers/thenables.js":[function(require,module,exports){
(function (global){
"use strict";

var adapter = global.adapter;
var resolved = adapter.resolved;
var rejected = adapter.rejected;
var deferred = adapter.deferred;

var other = { other: "other" }; // a value we don't want to be strict equal to

exports.fulfilled = {
    "a synchronously-fulfilled custom thenable": function (value) {
        return {
            then: function (onFulfilled) {
                onFulfilled(value);
            }
        };
    },

    "an asynchronously-fulfilled custom thenable": function (value) {
        return {
            then: function (onFulfilled) {
                setTimeout(function () {
                    onFulfilled(value);
                }, 0);
            }
        };
    },

    "a synchronously-fulfilled one-time thenable": function (value) {
        var numberOfTimesThenRetrieved = 0;
        return Object.create(null, {
            then: {
                get: function () {
                    if (numberOfTimesThenRetrieved === 0) {
                        ++numberOfTimesThenRetrieved;
                        return function (onFulfilled) {
                            onFulfilled(value);
                        };
                    }
                    return null;
                }
            }
        });
    },

    "a thenable that tries to fulfill twice": function (value) {
        return {
            then: function (onFulfilled) {
                onFulfilled(value);
                onFulfilled(other);
            }
        };
    },

    "a thenable that fulfills but then throws": function (value) {
        return {
            then: function (onFulfilled) {
                onFulfilled(value);
                throw other;
            }
        };
    },

    "an already-fulfilled promise": function (value) {
        return resolved(value);
    },

    "an eventually-fulfilled promise": function (value) {
        var d = deferred();
        setTimeout(function () {
            d.resolve(value);
        }, 50);
        return d.promise;
    }
};

exports.rejected = {
    "a synchronously-rejected custom thenable": function (reason) {
        return {
            then: function (onFulfilled, onRejected) {
                onRejected(reason);
            }
        };
    },

    "an asynchronously-rejected custom thenable": function (reason) {
        return {
            then: function (onFulfilled, onRejected) {
                setTimeout(function () {
                    onRejected(reason);
                }, 0);
            }
        };
    },

    "a synchronously-rejected one-time thenable": function (reason) {
        var numberOfTimesThenRetrieved = 0;
        return Object.create(null, {
            then: {
                get: function () {
                    if (numberOfTimesThenRetrieved === 0) {
                        ++numberOfTimesThenRetrieved;
                        return function (onFulfilled, onRejected) {
                            onRejected(reason);
                        };
                    }
                    return null;
                }
            }
        });
    },

    "a thenable that immediately throws in `then`": function (reason) {
        return {
            then: function () {
                throw reason;
            }
        };
    },

    "an object with a throwing `then` accessor": function (reason) {
        return Object.create(null, {
            then: {
                get: function () {
                    throw reason;
                }
            }
        });
    },

    "an already-rejected promise": function (reason) {
        return rejected(reason);
    },

    "an eventually-rejected promise": function (reason) {
        var d = deferred();
        setTimeout(function () {
            d.reject(reason);
        }, 50);
        return d.promise;
    }
};

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon.js":[function(require,module,exports){
/**
 * Sinon core utilities. For internal use only.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2013 Christian Johansen
 */
"use strict";

var sinon = (function () {
    var sinon;
    var isNode = typeof module !== "undefined" && module.exports && typeof require === "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        sinon = module.exports = require("./sinon/util/core");
        require("./sinon/extend");
        require("./sinon/typeOf");
        require("./sinon/times_in_words");
        require("./sinon/spy");
        require("./sinon/call");
        require("./sinon/behavior");
        require("./sinon/stub");
        require("./sinon/mock");
        require("./sinon/collection");
        require("./sinon/assert");
        require("./sinon/sandbox");
        require("./sinon/test");
        require("./sinon/test_case");
        require("./sinon/match");
        require("./sinon/format");
        require("./sinon/log_error");
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
        sinon = module.exports;
    } else {
        sinon = {};
    }

    return sinon;
}());

},{"./sinon/assert":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/assert.js","./sinon/behavior":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/behavior.js","./sinon/call":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/call.js","./sinon/collection":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/collection.js","./sinon/extend":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/extend.js","./sinon/format":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/format.js","./sinon/log_error":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/log_error.js","./sinon/match":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/match.js","./sinon/mock":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/mock.js","./sinon/sandbox":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/sandbox.js","./sinon/spy":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/spy.js","./sinon/stub":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/stub.js","./sinon/test":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/test.js","./sinon/test_case":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/test_case.js","./sinon/times_in_words":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/times_in_words.js","./sinon/typeOf":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/typeOf.js","./sinon/util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/assert.js":[function(require,module,exports){
(function (global){
/**
 * @depend times_in_words.js
 * @depend util/core.js
 * @depend stub.js
 * @depend format.js
 */
/**
 * Assertions matching the test spy retrieval interface.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2013 Christian Johansen
 */
"use strict";

(function (sinon, global) {
    var slice = Array.prototype.slice;

    function makeApi(sinon) {
        var assert;

        function verifyIsStub() {
            var method;

            for (var i = 0, l = arguments.length; i < l; ++i) {
                method = arguments[i];

                if (!method) {
                    assert.fail("fake is not a spy");
                }

                if (typeof method != "function") {
                    assert.fail(method + " is not a function");
                }

                if (typeof method.getCall != "function") {
                    assert.fail(method + " is not stubbed");
                }
            }
        }

        function failAssertion(object, msg) {
            object = object || global;
            var failMethod = object.fail || assert.fail;
            failMethod.call(object, msg);
        }

        function mirrorPropAsAssertion(name, method, message) {
            if (arguments.length == 2) {
                message = method;
                method = name;
            }

            assert[name] = function (fake) {
                verifyIsStub(fake);

                var args = slice.call(arguments, 1);
                var failed = false;

                if (typeof method == "function") {
                    failed = !method(fake);
                } else {
                    failed = typeof fake[method] == "function" ?
                        !fake[method].apply(fake, args) : !fake[method];
                }

                if (failed) {
                    failAssertion(this, fake.printf.apply(fake, [message].concat(args)));
                } else {
                    assert.pass(name);
                }
            };
        }

        function exposedName(prefix, prop) {
            return !prefix || /^fail/.test(prop) ? prop :
                prefix + prop.slice(0, 1).toUpperCase() + prop.slice(1);
        }

        assert = {
            failException: "AssertError",

            fail: function fail(message) {
                var error = new Error(message);
                error.name = this.failException || assert.failException;

                throw error;
            },

            pass: function pass(assertion) {},

            callOrder: function assertCallOrder() {
                verifyIsStub.apply(null, arguments);
                var expected = "", actual = "";

                if (!sinon.calledInOrder(arguments)) {
                    try {
                        expected = [].join.call(arguments, ", ");
                        var calls = slice.call(arguments);
                        var i = calls.length;
                        while (i) {
                            if (!calls[--i].called) {
                                calls.splice(i, 1);
                            }
                        }
                        actual = sinon.orderByFirstCall(calls).join(", ");
                    } catch (e) {
                        // If this fails, we'll just fall back to the blank string
                    }

                    failAssertion(this, "expected " + expected + " to be " +
                                "called in order but were called as " + actual);
                } else {
                    assert.pass("callOrder");
                }
            },

            callCount: function assertCallCount(method, count) {
                verifyIsStub(method);

                if (method.callCount != count) {
                    var msg = "expected %n to be called " + sinon.timesInWords(count) +
                        " but was called %c%C";
                    failAssertion(this, method.printf(msg));
                } else {
                    assert.pass("callCount");
                }
            },

            expose: function expose(target, options) {
                if (!target) {
                    throw new TypeError("target is null or undefined");
                }

                var o = options || {};
                var prefix = typeof o.prefix == "undefined" && "assert" || o.prefix;
                var includeFail = typeof o.includeFail == "undefined" || !!o.includeFail;

                for (var method in this) {
                    if (method != "expose" && (includeFail || !/^(fail)/.test(method))) {
                        target[exposedName(prefix, method)] = this[method];
                    }
                }

                return target;
            },

            match: function match(actual, expectation) {
                var matcher = sinon.match(expectation);
                if (matcher.test(actual)) {
                    assert.pass("match");
                } else {
                    var formatted = [
                        "expected value to match",
                        "    expected = " + sinon.format(expectation),
                        "    actual = " + sinon.format(actual)
                    ]
                    failAssertion(this, formatted.join("\n"));
                }
            }
        };

        mirrorPropAsAssertion("called", "expected %n to have been called at least once but was never called");
        mirrorPropAsAssertion("notCalled", function (spy) { return !spy.called; },
                            "expected %n to not have been called but was called %c%C");
        mirrorPropAsAssertion("calledOnce", "expected %n to be called once but was called %c%C");
        mirrorPropAsAssertion("calledTwice", "expected %n to be called twice but was called %c%C");
        mirrorPropAsAssertion("calledThrice", "expected %n to be called thrice but was called %c%C");
        mirrorPropAsAssertion("calledOn", "expected %n to be called with %1 as this but was called with %t");
        mirrorPropAsAssertion("alwaysCalledOn", "expected %n to always be called with %1 as this but was called with %t");
        mirrorPropAsAssertion("calledWithNew", "expected %n to be called with new");
        mirrorPropAsAssertion("alwaysCalledWithNew", "expected %n to always be called with new");
        mirrorPropAsAssertion("calledWith", "expected %n to be called with arguments %*%C");
        mirrorPropAsAssertion("calledWithMatch", "expected %n to be called with match %*%C");
        mirrorPropAsAssertion("alwaysCalledWith", "expected %n to always be called with arguments %*%C");
        mirrorPropAsAssertion("alwaysCalledWithMatch", "expected %n to always be called with match %*%C");
        mirrorPropAsAssertion("calledWithExactly", "expected %n to be called with exact arguments %*%C");
        mirrorPropAsAssertion("alwaysCalledWithExactly", "expected %n to always be called with exact arguments %*%C");
        mirrorPropAsAssertion("neverCalledWith", "expected %n to never be called with arguments %*%C");
        mirrorPropAsAssertion("neverCalledWithMatch", "expected %n to never be called with match %*%C");
        mirrorPropAsAssertion("threw", "%n did not throw exception%C");
        mirrorPropAsAssertion("alwaysThrew", "%n did not always throw exception%C");

        sinon.assert = assert;
        return assert;
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        require("./match");
        module.exports = makeApi(sinon);
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }

}(typeof sinon == "object" && sinon || null, typeof window != "undefined" ? window : (typeof self != "undefined") ? self : global));

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./match":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/match.js","./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/behavior.js":[function(require,module,exports){
(function (process){
/**
 * @depend util/core.js
 * @depend extend.js
 */
/**
 * Stub behavior
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @author Tim Fischbach (mail@timfischbach.de)
 * @license BSD
 *
 * Copyright (c) 2010-2013 Christian Johansen
 */
"use strict";

(function (sinon) {
    var slice = Array.prototype.slice;
    var join = Array.prototype.join;

    var nextTick = (function () {
        if (typeof process === "object" && typeof process.nextTick === "function") {
            return process.nextTick;
        } else if (typeof setImmediate === "function") {
            return setImmediate;
        } else {
            return function (callback) {
                setTimeout(callback, 0);
            };
        }
    })();

    function throwsException(error, message) {
        if (typeof error == "string") {
            this.exception = new Error(message || "");
            this.exception.name = error;
        } else if (!error) {
            this.exception = new Error("Error");
        } else {
            this.exception = error;
        }

        return this;
    }

    function getCallback(behavior, args) {
        var callArgAt = behavior.callArgAt;

        if (callArgAt < 0) {
            var callArgProp = behavior.callArgProp;

            for (var i = 0, l = args.length; i < l; ++i) {
                if (!callArgProp && typeof args[i] == "function") {
                    return args[i];
                }

                if (callArgProp && args[i] &&
                    typeof args[i][callArgProp] == "function") {
                    return args[i][callArgProp];
                }
            }

            return null;
        }

        return args[callArgAt];
    }

    function makeApi(sinon) {
        function getCallbackError(behavior, func, args) {
            if (behavior.callArgAt < 0) {
                var msg;

                if (behavior.callArgProp) {
                    msg = sinon.functionName(behavior.stub) +
                        " expected to yield to '" + behavior.callArgProp +
                        "', but no object with such a property was passed.";
                } else {
                    msg = sinon.functionName(behavior.stub) +
                        " expected to yield, but no callback was passed.";
                }

                if (args.length > 0) {
                    msg += " Received [" + join.call(args, ", ") + "]";
                }

                return msg;
            }

            return "argument at index " + behavior.callArgAt + " is not a function: " + func;
        }

        function callCallback(behavior, args) {
            if (typeof behavior.callArgAt == "number") {
                var func = getCallback(behavior, args);

                if (typeof func != "function") {
                    throw new TypeError(getCallbackError(behavior, func, args));
                }

                if (behavior.callbackAsync) {
                    nextTick(function () {
                        func.apply(behavior.callbackContext, behavior.callbackArguments);
                    });
                } else {
                    func.apply(behavior.callbackContext, behavior.callbackArguments);
                }
            }
        }

        var proto = {
            create: function create(stub) {
                var behavior = sinon.extend({}, sinon.behavior);
                delete behavior.create;
                behavior.stub = stub;

                return behavior;
            },

            isPresent: function isPresent() {
                return (typeof this.callArgAt == "number" ||
                        this.exception ||
                        typeof this.returnArgAt == "number" ||
                        this.returnThis ||
                        this.returnValueDefined);
            },

            invoke: function invoke(context, args) {
                callCallback(this, args);

                if (this.exception) {
                    throw this.exception;
                } else if (typeof this.returnArgAt == "number") {
                    return args[this.returnArgAt];
                } else if (this.returnThis) {
                    return context;
                }

                return this.returnValue;
            },

            onCall: function onCall(index) {
                return this.stub.onCall(index);
            },

            onFirstCall: function onFirstCall() {
                return this.stub.onFirstCall();
            },

            onSecondCall: function onSecondCall() {
                return this.stub.onSecondCall();
            },

            onThirdCall: function onThirdCall() {
                return this.stub.onThirdCall();
            },

            withArgs: function withArgs(/* arguments */) {
                throw new Error("Defining a stub by invoking \"stub.onCall(...).withArgs(...)\" is not supported. " +
                                "Use \"stub.withArgs(...).onCall(...)\" to define sequential behavior for calls with certain arguments.");
            },

            callsArg: function callsArg(pos) {
                if (typeof pos != "number") {
                    throw new TypeError("argument index is not number");
                }

                this.callArgAt = pos;
                this.callbackArguments = [];
                this.callbackContext = undefined;
                this.callArgProp = undefined;
                this.callbackAsync = false;

                return this;
            },

            callsArgOn: function callsArgOn(pos, context) {
                if (typeof pos != "number") {
                    throw new TypeError("argument index is not number");
                }
                if (typeof context != "object") {
                    throw new TypeError("argument context is not an object");
                }

                this.callArgAt = pos;
                this.callbackArguments = [];
                this.callbackContext = context;
                this.callArgProp = undefined;
                this.callbackAsync = false;

                return this;
            },

            callsArgWith: function callsArgWith(pos) {
                if (typeof pos != "number") {
                    throw new TypeError("argument index is not number");
                }

                this.callArgAt = pos;
                this.callbackArguments = slice.call(arguments, 1);
                this.callbackContext = undefined;
                this.callArgProp = undefined;
                this.callbackAsync = false;

                return this;
            },

            callsArgOnWith: function callsArgWith(pos, context) {
                if (typeof pos != "number") {
                    throw new TypeError("argument index is not number");
                }
                if (typeof context != "object") {
                    throw new TypeError("argument context is not an object");
                }

                this.callArgAt = pos;
                this.callbackArguments = slice.call(arguments, 2);
                this.callbackContext = context;
                this.callArgProp = undefined;
                this.callbackAsync = false;

                return this;
            },

            yields: function () {
                this.callArgAt = -1;
                this.callbackArguments = slice.call(arguments, 0);
                this.callbackContext = undefined;
                this.callArgProp = undefined;
                this.callbackAsync = false;

                return this;
            },

            yieldsOn: function (context) {
                if (typeof context != "object") {
                    throw new TypeError("argument context is not an object");
                }

                this.callArgAt = -1;
                this.callbackArguments = slice.call(arguments, 1);
                this.callbackContext = context;
                this.callArgProp = undefined;
                this.callbackAsync = false;

                return this;
            },

            yieldsTo: function (prop) {
                this.callArgAt = -1;
                this.callbackArguments = slice.call(arguments, 1);
                this.callbackContext = undefined;
                this.callArgProp = prop;
                this.callbackAsync = false;

                return this;
            },

            yieldsToOn: function (prop, context) {
                if (typeof context != "object") {
                    throw new TypeError("argument context is not an object");
                }

                this.callArgAt = -1;
                this.callbackArguments = slice.call(arguments, 2);
                this.callbackContext = context;
                this.callArgProp = prop;
                this.callbackAsync = false;

                return this;
            },

            throws: throwsException,
            throwsException: throwsException,

            returns: function returns(value) {
                this.returnValue = value;
                this.returnValueDefined = true;

                return this;
            },

            returnsArg: function returnsArg(pos) {
                if (typeof pos != "number") {
                    throw new TypeError("argument index is not number");
                }

                this.returnArgAt = pos;

                return this;
            },

            returnsThis: function returnsThis() {
                this.returnThis = true;

                return this;
            }
        };

        // create asynchronous versions of callsArg* and yields* methods
        for (var method in proto) {
            // need to avoid creating anotherasync versions of the newly added async methods
            if (proto.hasOwnProperty(method) &&
                method.match(/^(callsArg|yields)/) &&
                !method.match(/Async/)) {
                proto[method + "Async"] = (function (syncFnName) {
                    return function () {
                        var result = this[syncFnName].apply(this, arguments);
                        this.callbackAsync = true;
                        return result;
                    };
                })(method);
            }
        }

        sinon.behavior = proto;
        return proto;
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        module.exports = makeApi(sinon);
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(typeof sinon == "object" && sinon || null));

}).call(this,require('_process'))
},{"./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js","_process":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/process/browser.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/call.js":[function(require,module,exports){
/**
  * @depend util/core.js
  * @depend match.js
  * @depend format.js
  */
/**
  * Spy calls
  *
  * @author Christian Johansen (christian@cjohansen.no)
  * @author Maximilian Antoni (mail@maxantoni.de)
  * @license BSD
  *
  * Copyright (c) 2010-2013 Christian Johansen
  * Copyright (c) 2013 Maximilian Antoni
  */
"use strict";

(function (sinon) {
    function makeApi(sinon) {
        function throwYieldError(proxy, text, args) {
            var msg = sinon.functionName(proxy) + text;
            if (args.length) {
                msg += " Received [" + slice.call(args).join(", ") + "]";
            }
            throw new Error(msg);
        }

        var slice = Array.prototype.slice;

        var callProto = {
            calledOn: function calledOn(thisValue) {
                if (sinon.match && sinon.match.isMatcher(thisValue)) {
                    return thisValue.test(this.thisValue);
                }
                return this.thisValue === thisValue;
            },

            calledWith: function calledWith() {
                for (var i = 0, l = arguments.length; i < l; i += 1) {
                    if (!sinon.deepEqual(arguments[i], this.args[i])) {
                        return false;
                    }
                }

                return true;
            },

            calledWithMatch: function calledWithMatch() {
                for (var i = 0, l = arguments.length; i < l; i += 1) {
                    var actual = this.args[i];
                    var expectation = arguments[i];
                    if (!sinon.match || !sinon.match(expectation).test(actual)) {
                        return false;
                    }
                }
                return true;
            },

            calledWithExactly: function calledWithExactly() {
                return arguments.length == this.args.length &&
                    this.calledWith.apply(this, arguments);
            },

            notCalledWith: function notCalledWith() {
                return !this.calledWith.apply(this, arguments);
            },

            notCalledWithMatch: function notCalledWithMatch() {
                return !this.calledWithMatch.apply(this, arguments);
            },

            returned: function returned(value) {
                return sinon.deepEqual(value, this.returnValue);
            },

            threw: function threw(error) {
                if (typeof error === "undefined" || !this.exception) {
                    return !!this.exception;
                }

                return this.exception === error || this.exception.name === error;
            },

            calledWithNew: function calledWithNew() {
                return this.proxy.prototype && this.thisValue instanceof this.proxy;
            },

            calledBefore: function (other) {
                return this.callId < other.callId;
            },

            calledAfter: function (other) {
                return this.callId > other.callId;
            },

            callArg: function (pos) {
                this.args[pos]();
            },

            callArgOn: function (pos, thisValue) {
                this.args[pos].apply(thisValue);
            },

            callArgWith: function (pos) {
                this.callArgOnWith.apply(this, [pos, null].concat(slice.call(arguments, 1)));
            },

            callArgOnWith: function (pos, thisValue) {
                var args = slice.call(arguments, 2);
                this.args[pos].apply(thisValue, args);
            },

            yield: function () {
                this.yieldOn.apply(this, [null].concat(slice.call(arguments, 0)));
            },

            yieldOn: function (thisValue) {
                var args = this.args;
                for (var i = 0, l = args.length; i < l; ++i) {
                    if (typeof args[i] === "function") {
                        args[i].apply(thisValue, slice.call(arguments, 1));
                        return;
                    }
                }
                throwYieldError(this.proxy, " cannot yield since no callback was passed.", args);
            },

            yieldTo: function (prop) {
                this.yieldToOn.apply(this, [prop, null].concat(slice.call(arguments, 1)));
            },

            yieldToOn: function (prop, thisValue) {
                var args = this.args;
                for (var i = 0, l = args.length; i < l; ++i) {
                    if (args[i] && typeof args[i][prop] === "function") {
                        args[i][prop].apply(thisValue, slice.call(arguments, 2));
                        return;
                    }
                }
                throwYieldError(this.proxy, " cannot yield to '" + prop +
                    "' since no callback was passed.", args);
            },

            toString: function () {
                var callStr = this.proxy.toString() + "(";
                var args = [];

                for (var i = 0, l = this.args.length; i < l; ++i) {
                    args.push(sinon.format(this.args[i]));
                }

                callStr = callStr + args.join(", ") + ")";

                if (typeof this.returnValue != "undefined") {
                    callStr += " => " + sinon.format(this.returnValue);
                }

                if (this.exception) {
                    callStr += " !" + this.exception.name;

                    if (this.exception.message) {
                        callStr += "(" + this.exception.message + ")";
                    }
                }

                return callStr;
            }
        };

        callProto.invokeCallback = callProto.yield;

        function createSpyCall(spy, thisValue, args, returnValue, exception, id) {
            if (typeof id !== "number") {
                throw new TypeError("Call id is not a number");
            }
            var proxyCall = sinon.create(callProto);
            proxyCall.proxy = spy;
            proxyCall.thisValue = thisValue;
            proxyCall.args = args;
            proxyCall.returnValue = returnValue;
            proxyCall.exception = exception;
            proxyCall.callId = id;

            return proxyCall;
        }
        createSpyCall.toString = callProto.toString; // used by mocks

        sinon.spyCall = createSpyCall;
        return createSpyCall;
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        require("./match");
        module.exports = makeApi(sinon);
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(typeof sinon == "object" && sinon || null));

},{"./match":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/match.js","./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/collection.js":[function(require,module,exports){
/**
 * @depend util/core.js
 * @depend stub.js
 * @depend mock.js
 */
/**
 * Collections of stubs, spies and mocks.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2013 Christian Johansen
 */
"use strict";

(function (sinon) {
    var push = [].push;
    var hasOwnProperty = Object.prototype.hasOwnProperty;

    function getFakes(fakeCollection) {
        if (!fakeCollection.fakes) {
            fakeCollection.fakes = [];
        }

        return fakeCollection.fakes;
    }

    function each(fakeCollection, method) {
        var fakes = getFakes(fakeCollection);

        for (var i = 0, l = fakes.length; i < l; i += 1) {
            if (typeof fakes[i][method] == "function") {
                fakes[i][method]();
            }
        }
    }

    function compact(fakeCollection) {
        var fakes = getFakes(fakeCollection);
        var i = 0;
        while (i < fakes.length) {
            fakes.splice(i, 1);
        }
    }

    function makeApi(sinon) {
        var collection = {
            verify: function resolve() {
                each(this, "verify");
            },

            restore: function restore() {
                each(this, "restore");
                compact(this);
            },

            reset: function restore() {
                each(this, "reset");
            },

            verifyAndRestore: function verifyAndRestore() {
                var exception;

                try {
                    this.verify();
                } catch (e) {
                    exception = e;
                }

                this.restore();

                if (exception) {
                    throw exception;
                }
            },

            add: function add(fake) {
                push.call(getFakes(this), fake);
                return fake;
            },

            spy: function spy() {
                return this.add(sinon.spy.apply(sinon, arguments));
            },

            stub: function stub(object, property, value) {
                if (property) {
                    var original = object[property];

                    if (typeof original != "function") {
                        if (!hasOwnProperty.call(object, property)) {
                            throw new TypeError("Cannot stub non-existent own property " + property);
                        }

                        object[property] = value;

                        return this.add({
                            restore: function () {
                                object[property] = original;
                            }
                        });
                    }
                }
                if (!property && !!object && typeof object == "object") {
                    var stubbedObj = sinon.stub.apply(sinon, arguments);

                    for (var prop in stubbedObj) {
                        if (typeof stubbedObj[prop] === "function") {
                            this.add(stubbedObj[prop]);
                        }
                    }

                    return stubbedObj;
                }

                return this.add(sinon.stub.apply(sinon, arguments));
            },

            mock: function mock() {
                return this.add(sinon.mock.apply(sinon, arguments));
            },

            inject: function inject(obj) {
                var col = this;

                obj.spy = function () {
                    return col.spy.apply(col, arguments);
                };

                obj.stub = function () {
                    return col.stub.apply(col, arguments);
                };

                obj.mock = function () {
                    return col.mock.apply(col, arguments);
                };

                return obj;
            }
        };

        sinon.collection = collection;
        return collection;
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        require("./mock");
        require("./spy");
        require("./stub");
        module.exports = makeApi(sinon);
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(typeof sinon == "object" && sinon || null));

},{"./mock":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/mock.js","./spy":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/spy.js","./stub":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/stub.js","./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/extend.js":[function(require,module,exports){
/**
 * @depend ../sinon.js
 */
"use strict";

(function (sinon) {
    function makeApi(sinon) {

        // Adapted from https://developer.mozilla.org/en/docs/ECMAScript_DontEnum_attribute#JScript_DontEnum_Bug
        var hasDontEnumBug = (function () {
            var obj = {
                constructor: function () {
                    return "0";
                },
                toString: function () {
                    return "1";
                },
                valueOf: function () {
                    return "2";
                },
                toLocaleString: function () {
                    return "3";
                },
                prototype: function () {
                    return "4";
                },
                isPrototypeOf: function () {
                    return "5";
                },
                propertyIsEnumerable: function () {
                    return "6";
                },
                hasOwnProperty: function () {
                    return "7";
                },
                length: function () {
                    return "8";
                },
                unique: function () {
                    return "9"
                }
            };

            var result = [];
            for (var prop in obj) {
                result.push(obj[prop]());
            }
            return result.join("") !== "0123456789";
        })();

        /* Public: Extend target in place with all (own) properties from sources in-order. Thus, last source will
         *         override properties in previous sources.
         *
         * target - The Object to extend
         * sources - Objects to copy properties from.
         *
         * Returns the extended target
         */
        function extend(target /*, sources */) {
            var sources = Array.prototype.slice.call(arguments, 1),
                source, i, prop;

            for (i = 0; i < sources.length; i++) {
                source = sources[i];

                for (prop in source) {
                    if (source.hasOwnProperty(prop)) {
                        target[prop] = source[prop];
                    }
                }

                // Make sure we copy (own) toString method even when in JScript with DontEnum bug
                // See https://developer.mozilla.org/en/docs/ECMAScript_DontEnum_attribute#JScript_DontEnum_Bug
                if (hasDontEnumBug && source.hasOwnProperty("toString") && source.toString !== target.toString) {
                    target.toString = source.toString;
                }
            }

            return target;
        };

        sinon.extend = extend;
        return sinon.extend;
    }

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        module.exports = makeApi(sinon);
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(typeof sinon == "object" && sinon || null));

},{"./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/format.js":[function(require,module,exports){
/**
 * @depend ../sinon.js
 */
/**
 * Format functions
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2014 Christian Johansen
 */
"use strict";

(function (sinon, formatio) {
    function makeApi(sinon) {
        function valueFormatter(value) {
            return "" + value;
        }

        function getFormatioFormatter() {
            var formatter = formatio.configure({
                    quoteStrings: false,
                    limitChildrenCount: 250
                });

            function format() {
                return formatter.ascii.apply(formatter, arguments);
            };

            return format;
        }

        function getNodeFormatter(value) {
            function format(value) {
                return typeof value == "object" && value.toString === Object.prototype.toString ? util.inspect(value) : value;
            };

            try {
                var util = require("util");
            } catch (e) {
                /* Node, but no util module - would be very old, but better safe than sorry */
            }

            return util ? format : valueFormatter;
        }

        var isNode = typeof module !== "undefined" && module.exports && typeof require == "function",
            formatter;

        if (isNode) {
            try {
                formatio = require("formatio");
            } catch (e) {}
        }

        if (formatio) {
            formatter = getFormatioFormatter()
        } else if (isNode) {
            formatter = getNodeFormatter();
        } else {
            formatter = valueFormatter;
        }

        sinon.format = formatter;
        return sinon.format;
    }

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        module.exports = makeApi(sinon);
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(
    (typeof sinon == "object" && sinon || null),
    (typeof formatio == "object" && formatio)
));

},{"./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js","formatio":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/node_modules/formatio/lib/formatio.js","util":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/util/util.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/log_error.js":[function(require,module,exports){
/**
 * @depend ../sinon.js
 */
/**
 * Logs errors
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2014 Christian Johansen
 */
"use strict";

(function (sinon) {
    // cache a reference to setTimeout, so that our reference won't be stubbed out
    // when using fake timers and errors will still get logged
    // https://github.com/cjohansen/Sinon.JS/issues/381
    var realSetTimeout = setTimeout;

    function makeApi(sinon) {

        function log() {}

        function logError(label, err) {
            var msg = label + " threw exception: ";

            sinon.log(msg + "[" + err.name + "] " + err.message);

            if (err.stack) {
                sinon.log(err.stack);
            }

            logError.setTimeout(function () {
                err.message = msg + err.message;
                throw err;
            }, 0);
        };

        // wrap realSetTimeout with something we can stub in tests
        logError.setTimeout = function (func, timeout) {
            realSetTimeout(func, timeout);
        }

        var exports = {};
        exports.log = sinon.log = log;
        exports.logError = sinon.logError = logError;

        return exports;
    }

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        module.exports = makeApi(sinon);
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(typeof sinon == "object" && sinon || null));

},{"./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/match.js":[function(require,module,exports){
/**
 * @depend util/core.js
 * @depend typeOf.js
 */
/*jslint eqeqeq: false, onevar: false, plusplus: false*/
/*global module, require, sinon*/
/**
 * Match functions
 *
 * @author Maximilian Antoni (mail@maxantoni.de)
 * @license BSD
 *
 * Copyright (c) 2012 Maximilian Antoni
 */
"use strict";

(function (sinon) {
    function makeApi(sinon) {
        function assertType(value, type, name) {
            var actual = sinon.typeOf(value);
            if (actual !== type) {
                throw new TypeError("Expected type of " + name + " to be " +
                    type + ", but was " + actual);
            }
        }

        var matcher = {
            toString: function () {
                return this.message;
            }
        };

        function isMatcher(object) {
            return matcher.isPrototypeOf(object);
        }

        function matchObject(expectation, actual) {
            if (actual === null || actual === undefined) {
                return false;
            }
            for (var key in expectation) {
                if (expectation.hasOwnProperty(key)) {
                    var exp = expectation[key];
                    var act = actual[key];
                    if (match.isMatcher(exp)) {
                        if (!exp.test(act)) {
                            return false;
                        }
                    } else if (sinon.typeOf(exp) === "object") {
                        if (!matchObject(exp, act)) {
                            return false;
                        }
                    } else if (!sinon.deepEqual(exp, act)) {
                        return false;
                    }
                }
            }
            return true;
        }

        matcher.or = function (m2) {
            if (!arguments.length) {
                throw new TypeError("Matcher expected");
            } else if (!isMatcher(m2)) {
                m2 = match(m2);
            }
            var m1 = this;
            var or = sinon.create(matcher);
            or.test = function (actual) {
                return m1.test(actual) || m2.test(actual);
            };
            or.message = m1.message + ".or(" + m2.message + ")";
            return or;
        };

        matcher.and = function (m2) {
            if (!arguments.length) {
                throw new TypeError("Matcher expected");
            } else if (!isMatcher(m2)) {
                m2 = match(m2);
            }
            var m1 = this;
            var and = sinon.create(matcher);
            and.test = function (actual) {
                return m1.test(actual) && m2.test(actual);
            };
            and.message = m1.message + ".and(" + m2.message + ")";
            return and;
        };

        var match = function (expectation, message) {
            var m = sinon.create(matcher);
            var type = sinon.typeOf(expectation);
            switch (type) {
            case "object":
                if (typeof expectation.test === "function") {
                    m.test = function (actual) {
                        return expectation.test(actual) === true;
                    };
                    m.message = "match(" + sinon.functionName(expectation.test) + ")";
                    return m;
                }
                var str = [];
                for (var key in expectation) {
                    if (expectation.hasOwnProperty(key)) {
                        str.push(key + ": " + expectation[key]);
                    }
                }
                m.test = function (actual) {
                    return matchObject(expectation, actual);
                };
                m.message = "match(" + str.join(", ") + ")";
                break;
            case "number":
                m.test = function (actual) {
                    return expectation == actual;
                };
                break;
            case "string":
                m.test = function (actual) {
                    if (typeof actual !== "string") {
                        return false;
                    }
                    return actual.indexOf(expectation) !== -1;
                };
                m.message = "match(\"" + expectation + "\")";
                break;
            case "regexp":
                m.test = function (actual) {
                    if (typeof actual !== "string") {
                        return false;
                    }
                    return expectation.test(actual);
                };
                break;
            case "function":
                m.test = expectation;
                if (message) {
                    m.message = message;
                } else {
                    m.message = "match(" + sinon.functionName(expectation) + ")";
                }
                break;
            default:
                m.test = function (actual) {
                    return sinon.deepEqual(expectation, actual);
                };
            }
            if (!m.message) {
                m.message = "match(" + expectation + ")";
            }
            return m;
        };

        match.isMatcher = isMatcher;

        match.any = match(function () {
            return true;
        }, "any");

        match.defined = match(function (actual) {
            return actual !== null && actual !== undefined;
        }, "defined");

        match.truthy = match(function (actual) {
            return !!actual;
        }, "truthy");

        match.falsy = match(function (actual) {
            return !actual;
        }, "falsy");

        match.same = function (expectation) {
            return match(function (actual) {
                return expectation === actual;
            }, "same(" + expectation + ")");
        };

        match.typeOf = function (type) {
            assertType(type, "string", "type");
            return match(function (actual) {
                return sinon.typeOf(actual) === type;
            }, "typeOf(\"" + type + "\")");
        };

        match.instanceOf = function (type) {
            assertType(type, "function", "type");
            return match(function (actual) {
                return actual instanceof type;
            }, "instanceOf(" + sinon.functionName(type) + ")");
        };

        function createPropertyMatcher(propertyTest, messagePrefix) {
            return function (property, value) {
                assertType(property, "string", "property");
                var onlyProperty = arguments.length === 1;
                var message = messagePrefix + "(\"" + property + "\"";
                if (!onlyProperty) {
                    message += ", " + value;
                }
                message += ")";
                return match(function (actual) {
                    if (actual === undefined || actual === null ||
                            !propertyTest(actual, property)) {
                        return false;
                    }
                    return onlyProperty || sinon.deepEqual(value, actual[property]);
                }, message);
            };
        }

        match.has = createPropertyMatcher(function (actual, property) {
            if (typeof actual === "object") {
                return property in actual;
            }
            return actual[property] !== undefined;
        }, "has");

        match.hasOwn = createPropertyMatcher(function (actual, property) {
            return actual.hasOwnProperty(property);
        }, "hasOwn");

        match.bool = match.typeOf("boolean");
        match.number = match.typeOf("number");
        match.string = match.typeOf("string");
        match.object = match.typeOf("object");
        match.func = match.typeOf("function");
        match.array = match.typeOf("array");
        match.regexp = match.typeOf("regexp");
        match.date = match.typeOf("date");

        sinon.match = match;
        return match;
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        module.exports = makeApi(sinon);
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(typeof sinon == "object" && sinon || null));

},{"./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/mock.js":[function(require,module,exports){
/**
 * @depend times_in_words.js
 * @depend util/core.js
 * @depend extend.js
 * @depend stub.js
 * @depend format.js
 */
/**
 * Mock functions.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2013 Christian Johansen
 */
"use strict";

(function (sinon) {
    function makeApi(sinon) {
        var push = [].push;
        var match = sinon.match;

        function mock(object) {
            if (!object) {
                return sinon.expectation.create("Anonymous mock");
            }

            return mock.create(object);
        }

        function each(collection, callback) {
            if (!collection) {
                return;
            }

            for (var i = 0, l = collection.length; i < l; i += 1) {
                callback(collection[i]);
            }
        }

        sinon.extend(mock, {
            create: function create(object) {
                if (!object) {
                    throw new TypeError("object is null");
                }

                var mockObject = sinon.extend({}, mock);
                mockObject.object = object;
                delete mockObject.create;

                return mockObject;
            },

            expects: function expects(method) {
                if (!method) {
                    throw new TypeError("method is falsy");
                }

                if (!this.expectations) {
                    this.expectations = {};
                    this.proxies = [];
                }

                if (!this.expectations[method]) {
                    this.expectations[method] = [];
                    var mockObject = this;

                    sinon.wrapMethod(this.object, method, function () {
                        return mockObject.invokeMethod(method, this, arguments);
                    });

                    push.call(this.proxies, method);
                }

                var expectation = sinon.expectation.create(method);
                push.call(this.expectations[method], expectation);

                return expectation;
            },

            restore: function restore() {
                var object = this.object;

                each(this.proxies, function (proxy) {
                    if (typeof object[proxy].restore == "function") {
                        object[proxy].restore();
                    }
                });
            },

            verify: function verify() {
                var expectations = this.expectations || {};
                var messages = [], met = [];

                each(this.proxies, function (proxy) {
                    each(expectations[proxy], function (expectation) {
                        if (!expectation.met()) {
                            push.call(messages, expectation.toString());
                        } else {
                            push.call(met, expectation.toString());
                        }
                    });
                });

                this.restore();

                if (messages.length > 0) {
                    sinon.expectation.fail(messages.concat(met).join("\n"));
                } else if (met.length > 0) {
                    sinon.expectation.pass(messages.concat(met).join("\n"));
                }

                return true;
            },

            invokeMethod: function invokeMethod(method, thisValue, args) {
                var expectations = this.expectations && this.expectations[method];
                var length = expectations && expectations.length || 0, i;

                for (i = 0; i < length; i += 1) {
                    if (!expectations[i].met() &&
                        expectations[i].allowsCall(thisValue, args)) {
                        return expectations[i].apply(thisValue, args);
                    }
                }

                var messages = [], available, exhausted = 0;

                for (i = 0; i < length; i += 1) {
                    if (expectations[i].allowsCall(thisValue, args)) {
                        available = available || expectations[i];
                    } else {
                        exhausted += 1;
                    }
                    push.call(messages, "    " + expectations[i].toString());
                }

                if (exhausted === 0) {
                    return available.apply(thisValue, args);
                }

                messages.unshift("Unexpected call: " + sinon.spyCall.toString.call({
                    proxy: method,
                    args: args
                }));

                sinon.expectation.fail(messages.join("\n"));
            }
        });

        var times = sinon.timesInWords;
        var slice = Array.prototype.slice;

        function callCountInWords(callCount) {
            if (callCount == 0) {
                return "never called";
            } else {
                return "called " + times(callCount);
            }
        }

        function expectedCallCountInWords(expectation) {
            var min = expectation.minCalls;
            var max = expectation.maxCalls;

            if (typeof min == "number" && typeof max == "number") {
                var str = times(min);

                if (min != max) {
                    str = "at least " + str + " and at most " + times(max);
                }

                return str;
            }

            if (typeof min == "number") {
                return "at least " + times(min);
            }

            return "at most " + times(max);
        }

        function receivedMinCalls(expectation) {
            var hasMinLimit = typeof expectation.minCalls == "number";
            return !hasMinLimit || expectation.callCount >= expectation.minCalls;
        }

        function receivedMaxCalls(expectation) {
            if (typeof expectation.maxCalls != "number") {
                return false;
            }

            return expectation.callCount == expectation.maxCalls;
        }

        function verifyMatcher(possibleMatcher, arg) {
            if (match && match.isMatcher(possibleMatcher)) {
                return possibleMatcher.test(arg);
            } else {
                return true;
            }
        }

        sinon.expectation = {
            minCalls: 1,
            maxCalls: 1,

            create: function create(methodName) {
                var expectation = sinon.extend(sinon.stub.create(), sinon.expectation);
                delete expectation.create;
                expectation.method = methodName;

                return expectation;
            },

            invoke: function invoke(func, thisValue, args) {
                this.verifyCallAllowed(thisValue, args);

                return sinon.spy.invoke.apply(this, arguments);
            },

            atLeast: function atLeast(num) {
                if (typeof num != "number") {
                    throw new TypeError("'" + num + "' is not number");
                }

                if (!this.limitsSet) {
                    this.maxCalls = null;
                    this.limitsSet = true;
                }

                this.minCalls = num;

                return this;
            },

            atMost: function atMost(num) {
                if (typeof num != "number") {
                    throw new TypeError("'" + num + "' is not number");
                }

                if (!this.limitsSet) {
                    this.minCalls = null;
                    this.limitsSet = true;
                }

                this.maxCalls = num;

                return this;
            },

            never: function never() {
                return this.exactly(0);
            },

            once: function once() {
                return this.exactly(1);
            },

            twice: function twice() {
                return this.exactly(2);
            },

            thrice: function thrice() {
                return this.exactly(3);
            },

            exactly: function exactly(num) {
                if (typeof num != "number") {
                    throw new TypeError("'" + num + "' is not a number");
                }

                this.atLeast(num);
                return this.atMost(num);
            },

            met: function met() {
                return !this.failed && receivedMinCalls(this);
            },

            verifyCallAllowed: function verifyCallAllowed(thisValue, args) {
                if (receivedMaxCalls(this)) {
                    this.failed = true;
                    sinon.expectation.fail(this.method + " already called " + times(this.maxCalls));
                }

                if ("expectedThis" in this && this.expectedThis !== thisValue) {
                    sinon.expectation.fail(this.method + " called with " + thisValue + " as thisValue, expected " +
                        this.expectedThis);
                }

                if (!("expectedArguments" in this)) {
                    return;
                }

                if (!args) {
                    sinon.expectation.fail(this.method + " received no arguments, expected " +
                        sinon.format(this.expectedArguments));
                }

                if (args.length < this.expectedArguments.length) {
                    sinon.expectation.fail(this.method + " received too few arguments (" + sinon.format(args) +
                        "), expected " + sinon.format(this.expectedArguments));
                }

                if (this.expectsExactArgCount &&
                    args.length != this.expectedArguments.length) {
                    sinon.expectation.fail(this.method + " received too many arguments (" + sinon.format(args) +
                        "), expected " + sinon.format(this.expectedArguments));
                }

                for (var i = 0, l = this.expectedArguments.length; i < l; i += 1) {

                    if (!verifyMatcher(this.expectedArguments[i], args[i])) {
                        sinon.expectation.fail(this.method + " received wrong arguments " + sinon.format(args) +
                            ", didn't match " + this.expectedArguments.toString());
                    }

                    if (!sinon.deepEqual(this.expectedArguments[i], args[i])) {
                        sinon.expectation.fail(this.method + " received wrong arguments " + sinon.format(args) +
                            ", expected " + sinon.format(this.expectedArguments));
                    }
                }
            },

            allowsCall: function allowsCall(thisValue, args) {
                if (this.met() && receivedMaxCalls(this)) {
                    return false;
                }

                if ("expectedThis" in this && this.expectedThis !== thisValue) {
                    return false;
                }

                if (!("expectedArguments" in this)) {
                    return true;
                }

                args = args || [];

                if (args.length < this.expectedArguments.length) {
                    return false;
                }

                if (this.expectsExactArgCount &&
                    args.length != this.expectedArguments.length) {
                    return false;
                }

                for (var i = 0, l = this.expectedArguments.length; i < l; i += 1) {
                    if (!verifyMatcher(this.expectedArguments[i], args[i])) {
                        return false;
                    }

                    if (!sinon.deepEqual(this.expectedArguments[i], args[i])) {
                        return false;
                    }
                }

                return true;
            },

            withArgs: function withArgs() {
                this.expectedArguments = slice.call(arguments);
                return this;
            },

            withExactArgs: function withExactArgs() {
                this.withArgs.apply(this, arguments);
                this.expectsExactArgCount = true;
                return this;
            },

            on: function on(thisValue) {
                this.expectedThis = thisValue;
                return this;
            },

            toString: function () {
                var args = (this.expectedArguments || []).slice();

                if (!this.expectsExactArgCount) {
                    push.call(args, "[...]");
                }

                var callStr = sinon.spyCall.toString.call({
                    proxy: this.method || "anonymous mock expectation",
                    args: args
                });

                var message = callStr.replace(", [...", "[, ...") + " " +
                    expectedCallCountInWords(this);

                if (this.met()) {
                    return "Expectation met: " + message;
                }

                return "Expected " + message + " (" +
                    callCountInWords(this.callCount) + ")";
            },

            verify: function verify() {
                if (!this.met()) {
                    sinon.expectation.fail(this.toString());
                } else {
                    sinon.expectation.pass(this.toString());
                }

                return true;
            },

            pass: function pass(message) {
                sinon.assert.pass(message);
            },

            fail: function fail(message) {
                var exception = new Error(message);
                exception.name = "ExpectationError";

                throw exception;
            }
        };

        sinon.mock = mock;
        return mock;
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        require("./call");
        require("./match");
        require("./spy");
        module.exports = makeApi(sinon);
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(typeof sinon == "object" && sinon || null));

},{"./call":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/call.js","./match":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/match.js","./spy":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/spy.js","./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/sandbox.js":[function(require,module,exports){
/**
 * @depend util/core.js
 * @depend extend.js
 * @depend collection.js
 * @depend util/fake_timers.js
 * @depend util/fake_server_with_clock.js
 */
/**
 * Manages fake collections as well as fake utilities such as Sinon's
 * timers and fake XHR implementation in one convenient object.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2013 Christian Johansen
 */
"use strict";

(function () {
    function makeApi(sinon) {
        var push = [].push;

        function exposeValue(sandbox, config, key, value) {
            if (!value) {
                return;
            }

            if (config.injectInto && !(key in config.injectInto)) {
                config.injectInto[key] = value;
                sandbox.injectedKeys.push(key);
            } else {
                push.call(sandbox.args, value);
            }
        }

        function prepareSandboxFromConfig(config) {
            var sandbox = sinon.create(sinon.sandbox);

            if (config.useFakeServer) {
                if (typeof config.useFakeServer == "object") {
                    sandbox.serverPrototype = config.useFakeServer;
                }

                sandbox.useFakeServer();
            }

            if (config.useFakeTimers) {
                if (typeof config.useFakeTimers == "object") {
                    sandbox.useFakeTimers.apply(sandbox, config.useFakeTimers);
                } else {
                    sandbox.useFakeTimers();
                }
            }

            return sandbox;
        }

        sinon.sandbox = sinon.extend(sinon.create(sinon.collection), {
            useFakeTimers: function useFakeTimers() {
                this.clock = sinon.useFakeTimers.apply(sinon, arguments);

                return this.add(this.clock);
            },

            serverPrototype: sinon.fakeServer,

            useFakeServer: function useFakeServer() {
                var proto = this.serverPrototype || sinon.fakeServer;

                if (!proto || !proto.create) {
                    return null;
                }

                this.server = proto.create();
                return this.add(this.server);
            },

            inject: function (obj) {
                sinon.collection.inject.call(this, obj);

                if (this.clock) {
                    obj.clock = this.clock;
                }

                if (this.server) {
                    obj.server = this.server;
                    obj.requests = this.server.requests;
                }

                obj.match = sinon.match;

                return obj;
            },

            restore: function () {
                sinon.collection.restore.apply(this, arguments);
                this.restoreContext();
            },

            restoreContext: function () {
                if (this.injectedKeys) {
                    for (var i = 0, j = this.injectedKeys.length; i < j; i++) {
                        delete this.injectInto[this.injectedKeys[i]];
                    }
                    this.injectedKeys = [];
                }
            },

            create: function (config) {
                if (!config) {
                    return sinon.create(sinon.sandbox);
                }

                var sandbox = prepareSandboxFromConfig(config);
                sandbox.args = sandbox.args || [];
                sandbox.injectedKeys = [];
                sandbox.injectInto = config.injectInto;
                var prop, value, exposed = sandbox.inject({});

                if (config.properties) {
                    for (var i = 0, l = config.properties.length; i < l; i++) {
                        prop = config.properties[i];
                        value = exposed[prop] || prop == "sandbox" && sandbox;
                        exposeValue(sandbox, config, prop, value);
                    }
                } else {
                    exposeValue(sandbox, config, "sandbox", value);
                }

                return sandbox;
            },

            match: sinon.match
        });

        sinon.sandbox.useFakeXMLHttpRequest = sinon.sandbox.useFakeServer;

        return sinon.sandbox;
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        require("./util/fake_server");
        require("./util/fake_timers");
        require("./collection");
        module.exports = makeApi(sinon);
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}());

},{"./collection":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/collection.js","./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js","./util/fake_server":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/fake_server.js","./util/fake_timers":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/fake_timers.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/spy.js":[function(require,module,exports){
/**
  * @depend times_in_words.js
  * @depend util/core.js
  * @depend extend.js
  * @depend call.js
  * @depend format.js
  */
/**
  * Spy functions
  *
  * @author Christian Johansen (christian@cjohansen.no)
  * @license BSD
  *
  * Copyright (c) 2010-2013 Christian Johansen
  */
"use strict";

(function (sinon) {
    function makeApi(sinon) {
        var push = Array.prototype.push;
        var slice = Array.prototype.slice;
        var callId = 0;

        function spy(object, property) {
            if (!property && typeof object == "function") {
                return spy.create(object);
            }

            if (!object && !property) {
                return spy.create(function () { });
            }

            var method = object[property];
            return sinon.wrapMethod(object, property, spy.create(method));
        }

        function matchingFake(fakes, args, strict) {
            if (!fakes) {
                return;
            }

            for (var i = 0, l = fakes.length; i < l; i++) {
                if (fakes[i].matches(args, strict)) {
                    return fakes[i];
                }
            }
        }

        function incrementCallCount() {
            this.called = true;
            this.callCount += 1;
            this.notCalled = false;
            this.calledOnce = this.callCount == 1;
            this.calledTwice = this.callCount == 2;
            this.calledThrice = this.callCount == 3;
        }

        function createCallProperties() {
            this.firstCall = this.getCall(0);
            this.secondCall = this.getCall(1);
            this.thirdCall = this.getCall(2);
            this.lastCall = this.getCall(this.callCount - 1);
        }

        var vars = "a,b,c,d,e,f,g,h,i,j,k,l";
        function createProxy(func) {
            // Retain the function length:
            var p;
            if (func.length) {
                eval("p = (function proxy(" + vars.substring(0, func.length * 2 - 1) +
                    ") { return p.invoke(func, this, slice.call(arguments)); });");
            } else {
                p = function proxy() {
                    return p.invoke(func, this, slice.call(arguments));
                };
            }
            return p;
        }

        var uuid = 0;

        // Public API
        var spyApi = {
            reset: function () {
                if (this.invoking) {
                    var err = new Error("Cannot reset Sinon function while invoking it. " +
                                        "Move the call to .reset outside of the callback.");
                    err.name = "InvalidResetException";
                    throw err;
                }

                this.called = false;
                this.notCalled = true;
                this.calledOnce = false;
                this.calledTwice = false;
                this.calledThrice = false;
                this.callCount = 0;
                this.firstCall = null;
                this.secondCall = null;
                this.thirdCall = null;
                this.lastCall = null;
                this.args = [];
                this.returnValues = [];
                this.thisValues = [];
                this.exceptions = [];
                this.callIds = [];
                if (this.fakes) {
                    for (var i = 0; i < this.fakes.length; i++) {
                        this.fakes[i].reset();
                    }
                }
            },

            create: function create(func) {
                var name;

                if (typeof func != "function") {
                    func = function () { };
                } else {
                    name = sinon.functionName(func);
                }

                var proxy = createProxy(func);

                sinon.extend(proxy, spy);
                delete proxy.create;
                sinon.extend(proxy, func);

                proxy.reset();
                proxy.prototype = func.prototype;
                proxy.displayName = name || "spy";
                proxy.toString = sinon.functionToString;
                proxy.instantiateFake = sinon.spy.create;
                proxy.id = "spy#" + uuid++;

                return proxy;
            },

            invoke: function invoke(func, thisValue, args) {
                var matching = matchingFake(this.fakes, args);
                var exception, returnValue;

                incrementCallCount.call(this);
                push.call(this.thisValues, thisValue);
                push.call(this.args, args);
                push.call(this.callIds, callId++);

                // Make call properties available from within the spied function:
                createCallProperties.call(this);

                try {
                    this.invoking = true;

                    if (matching) {
                        returnValue = matching.invoke(func, thisValue, args);
                    } else {
                        returnValue = (this.func || func).apply(thisValue, args);
                    }

                    var thisCall = this.getCall(this.callCount - 1);
                    if (thisCall.calledWithNew() && typeof returnValue !== "object") {
                        returnValue = thisValue;
                    }
                } catch (e) {
                    exception = e;
                } finally {
                    delete this.invoking;
                }

                push.call(this.exceptions, exception);
                push.call(this.returnValues, returnValue);

                // Make return value and exception available in the calls:
                createCallProperties.call(this);

                if (exception !== undefined) {
                    throw exception;
                }

                return returnValue;
            },

            named: function named(name) {
                this.displayName = name;
                return this;
            },

            getCall: function getCall(i) {
                if (i < 0 || i >= this.callCount) {
                    return null;
                }

                return sinon.spyCall(this, this.thisValues[i], this.args[i],
                                        this.returnValues[i], this.exceptions[i],
                                        this.callIds[i]);
            },

            getCalls: function () {
                var calls = [];
                var i;

                for (i = 0; i < this.callCount; i++) {
                    calls.push(this.getCall(i));
                }

                return calls;
            },

            calledBefore: function calledBefore(spyFn) {
                if (!this.called) {
                    return false;
                }

                if (!spyFn.called) {
                    return true;
                }

                return this.callIds[0] < spyFn.callIds[spyFn.callIds.length - 1];
            },

            calledAfter: function calledAfter(spyFn) {
                if (!this.called || !spyFn.called) {
                    return false;
                }

                return this.callIds[this.callCount - 1] > spyFn.callIds[spyFn.callCount - 1];
            },

            withArgs: function () {
                var args = slice.call(arguments);

                if (this.fakes) {
                    var match = matchingFake(this.fakes, args, true);

                    if (match) {
                        return match;
                    }
                } else {
                    this.fakes = [];
                }

                var original = this;
                var fake = this.instantiateFake();
                fake.matchingAguments = args;
                fake.parent = this;
                push.call(this.fakes, fake);

                fake.withArgs = function () {
                    return original.withArgs.apply(original, arguments);
                };

                for (var i = 0; i < this.args.length; i++) {
                    if (fake.matches(this.args[i])) {
                        incrementCallCount.call(fake);
                        push.call(fake.thisValues, this.thisValues[i]);
                        push.call(fake.args, this.args[i]);
                        push.call(fake.returnValues, this.returnValues[i]);
                        push.call(fake.exceptions, this.exceptions[i]);
                        push.call(fake.callIds, this.callIds[i]);
                    }
                }
                createCallProperties.call(fake);

                return fake;
            },

            matches: function (args, strict) {
                var margs = this.matchingAguments;

                if (margs.length <= args.length &&
                    sinon.deepEqual(margs, args.slice(0, margs.length))) {
                    return !strict || margs.length == args.length;
                }
            },

            printf: function (format) {
                var spy = this;
                var args = slice.call(arguments, 1);
                var formatter;

                return (format || "").replace(/%(.)/g, function (match, specifyer) {
                    formatter = spyApi.formatters[specifyer];

                    if (typeof formatter == "function") {
                        return formatter.call(null, spy, args);
                    } else if (!isNaN(parseInt(specifyer, 10))) {
                        return sinon.format(args[specifyer - 1]);
                    }

                    return "%" + specifyer;
                });
            }
        };

        function delegateToCalls(method, matchAny, actual, notCalled) {
            spyApi[method] = function () {
                if (!this.called) {
                    if (notCalled) {
                        return notCalled.apply(this, arguments);
                    }
                    return false;
                }

                var currentCall;
                var matches = 0;

                for (var i = 0, l = this.callCount; i < l; i += 1) {
                    currentCall = this.getCall(i);

                    if (currentCall[actual || method].apply(currentCall, arguments)) {
                        matches += 1;

                        if (matchAny) {
                            return true;
                        }
                    }
                }

                return matches === this.callCount;
            };
        }

        delegateToCalls("calledOn", true);
        delegateToCalls("alwaysCalledOn", false, "calledOn");
        delegateToCalls("calledWith", true);
        delegateToCalls("calledWithMatch", true);
        delegateToCalls("alwaysCalledWith", false, "calledWith");
        delegateToCalls("alwaysCalledWithMatch", false, "calledWithMatch");
        delegateToCalls("calledWithExactly", true);
        delegateToCalls("alwaysCalledWithExactly", false, "calledWithExactly");
        delegateToCalls("neverCalledWith", false, "notCalledWith",
            function () { return true; });
        delegateToCalls("neverCalledWithMatch", false, "notCalledWithMatch",
            function () { return true; });
        delegateToCalls("threw", true);
        delegateToCalls("alwaysThrew", false, "threw");
        delegateToCalls("returned", true);
        delegateToCalls("alwaysReturned", false, "returned");
        delegateToCalls("calledWithNew", true);
        delegateToCalls("alwaysCalledWithNew", false, "calledWithNew");
        delegateToCalls("callArg", false, "callArgWith", function () {
            throw new Error(this.toString() + " cannot call arg since it was not yet invoked.");
        });
        spyApi.callArgWith = spyApi.callArg;
        delegateToCalls("callArgOn", false, "callArgOnWith", function () {
            throw new Error(this.toString() + " cannot call arg since it was not yet invoked.");
        });
        spyApi.callArgOnWith = spyApi.callArgOn;
        delegateToCalls("yield", false, "yield", function () {
            throw new Error(this.toString() + " cannot yield since it was not yet invoked.");
        });
        // "invokeCallback" is an alias for "yield" since "yield" is invalid in strict mode.
        spyApi.invokeCallback = spyApi.yield;
        delegateToCalls("yieldOn", false, "yieldOn", function () {
            throw new Error(this.toString() + " cannot yield since it was not yet invoked.");
        });
        delegateToCalls("yieldTo", false, "yieldTo", function (property) {
            throw new Error(this.toString() + " cannot yield to '" + property +
                "' since it was not yet invoked.");
        });
        delegateToCalls("yieldToOn", false, "yieldToOn", function (property) {
            throw new Error(this.toString() + " cannot yield to '" + property +
                "' since it was not yet invoked.");
        });

        spyApi.formatters = {
            c: function (spy) {
                return sinon.timesInWords(spy.callCount);
            },

            n: function (spy) {
                return spy.toString();
            },

            C: function (spy) {
                var calls = [];

                for (var i = 0, l = spy.callCount; i < l; ++i) {
                    var stringifiedCall = "    " + spy.getCall(i).toString();
                    if (/\n/.test(calls[i - 1])) {
                        stringifiedCall = "\n" + stringifiedCall;
                    }
                    push.call(calls, stringifiedCall);
                }

                return calls.length > 0 ? "\n" + calls.join("\n") : "";
            },

            t: function (spy) {
                var objects = [];

                for (var i = 0, l = spy.callCount; i < l; ++i) {
                    push.call(objects, sinon.format(spy.thisValues[i]));
                }

                return objects.join(", ");
            },

            "*": function (spy, args) {
                var formatted = [];

                for (var i = 0, l = args.length; i < l; ++i) {
                    push.call(formatted, sinon.format(args[i]));
                }

                return formatted.join(", ");
            }
        };

        sinon.extend(spy, spyApi);

        spy.spyCall = sinon.spyCall;
        sinon.spy = spy;

        return spy;
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        require("./call");
        module.exports = makeApi(sinon);
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(typeof sinon == "object" && sinon || null));

},{"./call":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/call.js","./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/stub.js":[function(require,module,exports){
/**
 * @depend util/core.js
 * @depend extend.js
 * @depend spy.js
 * @depend behavior.js
 */
/**
 * Stub functions
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2013 Christian Johansen
 */
"use strict";

(function (sinon) {
    function makeApi(sinon) {
        function stub(object, property, func) {
            if (!!func && typeof func != "function") {
                throw new TypeError("Custom stub should be function");
            }

            var wrapper;

            if (func) {
                wrapper = sinon.spy && sinon.spy.create ? sinon.spy.create(func) : func;
            } else {
                wrapper = stub.create();
            }

            if (!object && typeof property === "undefined") {
                return sinon.stub.create();
            }

            if (typeof property === "undefined" && typeof object == "object") {
                for (var prop in object) {
                    if (typeof object[prop] === "function") {
                        stub(object, prop);
                    }
                }

                return object;
            }

            return sinon.wrapMethod(object, property, wrapper);
        }

        function getDefaultBehavior(stub) {
            return stub.defaultBehavior || getParentBehaviour(stub) || sinon.behavior.create(stub);
        }

        function getParentBehaviour(stub) {
            return (stub.parent && getCurrentBehavior(stub.parent));
        }

        function getCurrentBehavior(stub) {
            var behavior = stub.behaviors[stub.callCount - 1];
            return behavior && behavior.isPresent() ? behavior : getDefaultBehavior(stub);
        }

        var uuid = 0;

        var proto = {
            create: function create() {
                var functionStub = function () {
                    return getCurrentBehavior(functionStub).invoke(this, arguments);
                };

                functionStub.id = "stub#" + uuid++;
                var orig = functionStub;
                functionStub = sinon.spy.create(functionStub);
                functionStub.func = orig;

                sinon.extend(functionStub, stub);
                functionStub.instantiateFake = sinon.stub.create;
                functionStub.displayName = "stub";
                functionStub.toString = sinon.functionToString;

                functionStub.defaultBehavior = null;
                functionStub.behaviors = [];

                return functionStub;
            },

            resetBehavior: function () {
                var i;

                this.defaultBehavior = null;
                this.behaviors = [];

                delete this.returnValue;
                delete this.returnArgAt;
                this.returnThis = false;

                if (this.fakes) {
                    for (i = 0; i < this.fakes.length; i++) {
                        this.fakes[i].resetBehavior();
                    }
                }
            },

            onCall: function onCall(index) {
                if (!this.behaviors[index]) {
                    this.behaviors[index] = sinon.behavior.create(this);
                }

                return this.behaviors[index];
            },

            onFirstCall: function onFirstCall() {
                return this.onCall(0);
            },

            onSecondCall: function onSecondCall() {
                return this.onCall(1);
            },

            onThirdCall: function onThirdCall() {
                return this.onCall(2);
            }
        };

        for (var method in sinon.behavior) {
            if (sinon.behavior.hasOwnProperty(method) &&
                !proto.hasOwnProperty(method) &&
                method != "create" &&
                method != "withArgs" &&
                method != "invoke") {
                proto[method] = (function (behaviorMethod) {
                    return function () {
                        this.defaultBehavior = this.defaultBehavior || sinon.behavior.create(this);
                        this.defaultBehavior[behaviorMethod].apply(this.defaultBehavior, arguments);
                        return this;
                    };
                }(method));
            }
        }

        sinon.extend(stub, proto);
        sinon.stub = stub;

        return stub;
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        require("./behavior");
        require("./spy");
        module.exports = makeApi(sinon);
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(typeof sinon == "object" && sinon || null));

},{"./behavior":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/behavior.js","./spy":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/spy.js","./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/test.js":[function(require,module,exports){
/**
 * @depend util/core.js
 * @depend stub.js
 * @depend mock.js
 * @depend sandbox.js
 */
/**
 * Test function, sandboxes fakes
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2013 Christian Johansen
 */
"use strict";

(function (sinon) {
    function makeApi(sinon) {
        function test(callback) {
            var type = typeof callback;

            if (type != "function") {
                throw new TypeError("sinon.test needs to wrap a test function, got " + type);
            }

            function sinonSandboxedTest() {
                var config = sinon.getConfig(sinon.config);
                config.injectInto = config.injectIntoThis && this || config.injectInto;
                var sandbox = sinon.sandbox.create(config);
                var exception, result;
                var doneIsWrapped = false;
                var argumentsCopy = Array.prototype.slice.call(arguments);
                if (argumentsCopy.length > 0 && typeof argumentsCopy[arguments.length - 1] == "function") {
                    var oldDone = argumentsCopy[arguments.length - 1];
                    argumentsCopy[arguments.length - 1] = function done(result) {
                        if (result) {
                            sandbox.restore();
                            throw exception;
                        } else {
                            sandbox.verifyAndRestore();
                        }
                        oldDone(result);
                    }
                    doneIsWrapped = true;
                }

                var args = argumentsCopy.concat(sandbox.args);

                try {
                    result = callback.apply(this, args);
                } catch (e) {
                    exception = e;
                }

                if (!doneIsWrapped) {
                    if (typeof exception !== "undefined") {
                        sandbox.restore();
                        throw exception;
                    } else {
                        sandbox.verifyAndRestore();
                    }
                }

                return result;
            };

            if (callback.length) {
                return function sinonAsyncSandboxedTest(callback) {
                    return sinonSandboxedTest.apply(this, arguments);
                };
            }

            return sinonSandboxedTest;
        }

        test.config = {
            injectIntoThis: true,
            injectInto: null,
            properties: ["spy", "stub", "mock", "clock", "server", "requests"],
            useFakeTimers: true,
            useFakeServer: true
        };

        sinon.test = test;
        return test;
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        require("./sandbox");
        module.exports = makeApi(sinon);
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(typeof sinon == "object" && sinon || null));

},{"./sandbox":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/sandbox.js","./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/test_case.js":[function(require,module,exports){
/**
 * @depend util/core.js
 * @depend test.js
 */
/**
 * Test case, sandboxes all test functions
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2013 Christian Johansen
 */
"use strict";

(function (sinon) {
    function createTest(property, setUp, tearDown) {
        return function () {
            if (setUp) {
                setUp.apply(this, arguments);
            }

            var exception, result;

            try {
                result = property.apply(this, arguments);
            } catch (e) {
                exception = e;
            }

            if (tearDown) {
                tearDown.apply(this, arguments);
            }

            if (exception) {
                throw exception;
            }

            return result;
        };
    }

    function makeApi(sinon) {
        function testCase(tests, prefix) {
            /*jsl:ignore*/
            if (!tests || typeof tests != "object") {
                throw new TypeError("sinon.testCase needs an object with test functions");
            }
            /*jsl:end*/

            prefix = prefix || "test";
            var rPrefix = new RegExp("^" + prefix);
            var methods = {}, testName, property, method;
            var setUp = tests.setUp;
            var tearDown = tests.tearDown;

            for (testName in tests) {
                if (tests.hasOwnProperty(testName)) {
                    property = tests[testName];

                    if (/^(setUp|tearDown)$/.test(testName)) {
                        continue;
                    }

                    if (typeof property == "function" && rPrefix.test(testName)) {
                        method = property;

                        if (setUp || tearDown) {
                            method = createTest(property, setUp, tearDown);
                        }

                        methods[testName] = sinon.test(method);
                    } else {
                        methods[testName] = tests[testName];
                    }
                }
            }

            return methods;
        }

        sinon.testCase = testCase;
        return testCase;
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        require("./test");
        module.exports = makeApi(sinon);
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(typeof sinon == "object" && sinon || null));

},{"./test":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/test.js","./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/times_in_words.js":[function(require,module,exports){
/**
 * @depend ../sinon.js
 */
"use strict";

(function (sinon) {
    function makeApi(sinon) {

        function timesInWords(count) {
            switch (count) {
                case 1:
                    return "once";
                case 2:
                    return "twice";
                case 3:
                    return "thrice";
                default:
                    return (count || 0) + " times";
            }
        }

        sinon.timesInWords = timesInWords;
        return sinon.timesInWords;
    }

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        module.exports = makeApi(sinon);
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(typeof sinon == "object" && sinon || null));

},{"./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/typeOf.js":[function(require,module,exports){
/**
 * @depend ../sinon.js
 */
/**
 * Format functions
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2014 Christian Johansen
 */
"use strict";

(function (sinon, formatio) {
    function makeApi(sinon) {
        function typeOf(value) {
            if (value === null) {
                return "null";
            } else if (value === undefined) {
                return "undefined";
            }
            var string = Object.prototype.toString.call(value);
            return string.substring(8, string.length - 1).toLowerCase();
        };

        sinon.typeOf = typeOf;
        return sinon.typeOf;
    }

    function loadDependencies(require, exports, module) {
        var sinon = require("./util/core");
        module.exports = makeApi(sinon);
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(
    (typeof sinon == "object" && sinon || null),
    (typeof formatio == "object" && formatio)
));

},{"./util/core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js":[function(require,module,exports){
/**
 * @depend ../../sinon.js
 */
/**
 * Sinon core utilities. For internal use only.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2013 Christian Johansen
 */
"use strict";

(function (sinon) {
    var div = typeof document != "undefined" && document.createElement("div");
    var hasOwn = Object.prototype.hasOwnProperty;

    function isDOMNode(obj) {
        var success = false;

        try {
            obj.appendChild(div);
            success = div.parentNode == obj;
        } catch (e) {
            return false;
        } finally {
            try {
                obj.removeChild(div);
            } catch (e) {
                // Remove failed, not much we can do about that
            }
        }

        return success;
    }

    function isElement(obj) {
        return div && obj && obj.nodeType === 1 && isDOMNode(obj);
    }

    function isFunction(obj) {
        return typeof obj === "function" || !!(obj && obj.constructor && obj.call && obj.apply);
    }

    function isReallyNaN(val) {
        return typeof val === "number" && isNaN(val);
    }

    function mirrorProperties(target, source) {
        for (var prop in source) {
            if (!hasOwn.call(target, prop)) {
                target[prop] = source[prop];
            }
        }
    }

    function isRestorable(obj) {
        return typeof obj === "function" && typeof obj.restore === "function" && obj.restore.sinon;
    }

    function makeApi(sinon) {
        sinon.wrapMethod = function wrapMethod(object, property, method) {
            if (!object) {
                throw new TypeError("Should wrap property of object");
            }

            if (typeof method != "function") {
                throw new TypeError("Method wrapper should be function");
            }

            var wrappedMethod = object[property],
                error;

            if (!isFunction(wrappedMethod)) {
                error = new TypeError("Attempted to wrap " + (typeof wrappedMethod) + " property " +
                                    property + " as function");
            } else if (wrappedMethod.restore && wrappedMethod.restore.sinon) {
                error = new TypeError("Attempted to wrap " + property + " which is already wrapped");
            } else if (wrappedMethod.calledBefore) {
                var verb = !!wrappedMethod.returns ? "stubbed" : "spied on";
                error = new TypeError("Attempted to wrap " + property + " which is already " + verb);
            }

            if (error) {
                if (wrappedMethod && wrappedMethod.stackTrace) {
                    error.stack += "\n--------------\n" + wrappedMethod.stackTrace;
                }
                throw error;
            }

            // IE 8 does not support hasOwnProperty on the window object and Firefox has a problem
            // when using hasOwn.call on objects from other frames.
            var owned = object.hasOwnProperty ? object.hasOwnProperty(property) : hasOwn.call(object, property);
            object[property] = method;
            method.displayName = property;
            // Set up a stack trace which can be used later to find what line of
            // code the original method was created on.
            method.stackTrace = (new Error("Stack Trace for original")).stack;

            method.restore = function () {
                // For prototype properties try to reset by delete first.
                // If this fails (ex: localStorage on mobile safari) then force a reset
                // via direct assignment.
                if (!owned) {
                    delete object[property];
                }
                if (object[property] === method) {
                    object[property] = wrappedMethod;
                }
            };

            method.restore.sinon = true;
            mirrorProperties(method, wrappedMethod);

            return method;
        };

        sinon.create = function create(proto) {
            var F = function () {};
            F.prototype = proto;
            return new F();
        };

        sinon.deepEqual = function deepEqual(a, b) {
            if (sinon.match && sinon.match.isMatcher(a)) {
                return a.test(b);
            }

            if (typeof a != "object" || typeof b != "object") {
                if (isReallyNaN(a) && isReallyNaN(b)) {
                    return true;
                } else {
                    return a === b;
                }
            }

            if (isElement(a) || isElement(b)) {
                return a === b;
            }

            if (a === b) {
                return true;
            }

            if ((a === null && b !== null) || (a !== null && b === null)) {
                return false;
            }

            if (a instanceof RegExp && b instanceof RegExp) {
                return (a.source === b.source) && (a.global === b.global) &&
                    (a.ignoreCase === b.ignoreCase) && (a.multiline === b.multiline);
            }

            var aString = Object.prototype.toString.call(a);
            if (aString != Object.prototype.toString.call(b)) {
                return false;
            }

            if (aString == "[object Date]") {
                return a.valueOf() === b.valueOf();
            }

            var prop, aLength = 0, bLength = 0;

            if (aString == "[object Array]" && a.length !== b.length) {
                return false;
            }

            for (prop in a) {
                aLength += 1;

                if (!(prop in b)) {
                    return false;
                }

                if (!deepEqual(a[prop], b[prop])) {
                    return false;
                }
            }

            for (prop in b) {
                bLength += 1;
            }

            return aLength == bLength;
        };

        sinon.functionName = function functionName(func) {
            var name = func.displayName || func.name;

            // Use function decomposition as a last resort to get function
            // name. Does not rely on function decomposition to work - if it
            // doesn't debugging will be slightly less informative
            // (i.e. toString will say 'spy' rather than 'myFunc').
            if (!name) {
                var matches = func.toString().match(/function ([^\s\(]+)/);
                name = matches && matches[1];
            }

            return name;
        };

        sinon.functionToString = function toString() {
            if (this.getCall && this.callCount) {
                var thisValue, prop, i = this.callCount;

                while (i--) {
                    thisValue = this.getCall(i).thisValue;

                    for (prop in thisValue) {
                        if (thisValue[prop] === this) {
                            return prop;
                        }
                    }
                }
            }

            return this.displayName || "sinon fake";
        };

        sinon.getConfig = function (custom) {
            var config = {};
            custom = custom || {};
            var defaults = sinon.defaultConfig;

            for (var prop in defaults) {
                if (defaults.hasOwnProperty(prop)) {
                    config[prop] = custom.hasOwnProperty(prop) ? custom[prop] : defaults[prop];
                }
            }

            return config;
        };

        sinon.defaultConfig = {
            injectIntoThis: true,
            injectInto: null,
            properties: ["spy", "stub", "mock", "clock", "server", "requests"],
            useFakeTimers: true,
            useFakeServer: true
        };

        sinon.timesInWords = function timesInWords(count) {
            return count == 1 && "once" ||
                count == 2 && "twice" ||
                count == 3 && "thrice" ||
                (count || 0) + " times";
        };

        sinon.calledInOrder = function (spies) {
            for (var i = 1, l = spies.length; i < l; i++) {
                if (!spies[i - 1].calledBefore(spies[i]) || !spies[i].called) {
                    return false;
                }
            }

            return true;
        };

        sinon.orderByFirstCall = function (spies) {
            return spies.sort(function (a, b) {
                // uuid, won't ever be equal
                var aCall = a.getCall(0);
                var bCall = b.getCall(0);
                var aId = aCall && aCall.callId || -1;
                var bId = bCall && bCall.callId || -1;

                return aId < bId ? -1 : 1;
            });
        };

        sinon.createStubInstance = function (constructor) {
            if (typeof constructor !== "function") {
                throw new TypeError("The constructor should be a function.");
            }
            return sinon.stub(sinon.create(constructor.prototype));
        };

        sinon.restore = function (object) {
            if (object !== null && typeof object === "object") {
                for (var prop in object) {
                    if (isRestorable(object[prop])) {
                        object[prop].restore();
                    }
                }
            } else if (isRestorable(object)) {
                object.restore();
            }
        };

        return sinon;
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports) {
        makeApi(exports);
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports);
    } else if (!sinon) {
        return;
    } else {
        makeApi(sinon);
    }
}(typeof sinon == "object" && sinon || null));

},{}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/event.js":[function(require,module,exports){
/**
 * Minimal Event interface implementation
 *
 * Original implementation by Sven Fuchs: https://gist.github.com/995028
 * Modifications and tests by Christian Johansen.
 *
 * @author Sven Fuchs (svenfuchs@artweb-design.de)
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2011 Sven Fuchs, Christian Johansen
 */
"use strict";

if (typeof sinon == "undefined") {
    this.sinon = {};
}

(function () {
    var push = [].push;

    function makeApi(sinon) {
        sinon.Event = function Event(type, bubbles, cancelable, target) {
            this.initEvent(type, bubbles, cancelable, target);
        };

        sinon.Event.prototype = {
            initEvent: function (type, bubbles, cancelable, target) {
                this.type = type;
                this.bubbles = bubbles;
                this.cancelable = cancelable;
                this.target = target;
            },

            stopPropagation: function () {},

            preventDefault: function () {
                this.defaultPrevented = true;
            }
        };

        sinon.ProgressEvent = function ProgressEvent(type, progressEventRaw, target) {
            this.initEvent(type, false, false, target);
            this.loaded = progressEventRaw.loaded || null;
            this.total = progressEventRaw.total || null;
        };

        sinon.ProgressEvent.prototype = new sinon.Event();

        sinon.ProgressEvent.prototype.constructor =  sinon.ProgressEvent;

        sinon.CustomEvent = function CustomEvent(type, customData, target) {
            this.initEvent(type, false, false, target);
            this.detail = customData.detail || null;
        };

        sinon.CustomEvent.prototype = new sinon.Event();

        sinon.CustomEvent.prototype.constructor =  sinon.CustomEvent;

        sinon.EventTarget = {
            addEventListener: function addEventListener(event, listener) {
                this.eventListeners = this.eventListeners || {};
                this.eventListeners[event] = this.eventListeners[event] || [];
                push.call(this.eventListeners[event], listener);
            },

            removeEventListener: function removeEventListener(event, listener) {
                var listeners = this.eventListeners && this.eventListeners[event] || [];

                for (var i = 0, l = listeners.length; i < l; ++i) {
                    if (listeners[i] == listener) {
                        return listeners.splice(i, 1);
                    }
                }
            },

            dispatchEvent: function dispatchEvent(event) {
                var type = event.type;
                var listeners = this.eventListeners && this.eventListeners[type] || [];

                for (var i = 0; i < listeners.length; i++) {
                    if (typeof listeners[i] == "function") {
                        listeners[i].call(this, event);
                    } else {
                        listeners[i].handleEvent(event);
                    }
                }

                return !!event.defaultPrevented;
            }
        };
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require) {
        var sinon = require("./core");
        makeApi(sinon);
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require);
    } else {
        makeApi(sinon);
    }
}());

},{"./core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/fake_server.js":[function(require,module,exports){
/**
 * @depend fake_xml_http_request.js
 * @depend ../format.js
 * @depend ../log_error.js
 */
/**
 * The Sinon "server" mimics a web server that receives requests from
 * sinon.FakeXMLHttpRequest and provides an API to respond to those requests,
 * both synchronously and asynchronously. To respond synchronuously, canned
 * answers have to be provided upfront.
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2013 Christian Johansen
 */
"use strict";

if (typeof sinon == "undefined") {
    var sinon = {};
}

(function () {
    var push = [].push;
    function F() {}

    function create(proto) {
        F.prototype = proto;
        return new F();
    }

    function responseArray(handler) {
        var response = handler;

        if (Object.prototype.toString.call(handler) != "[object Array]") {
            response = [200, {}, handler];
        }

        if (typeof response[2] != "string") {
            throw new TypeError("Fake server response body should be string, but was " +
                                typeof response[2]);
        }

        return response;
    }

    var wloc = typeof window !== "undefined" ? window.location : {};
    var rCurrLoc = new RegExp("^" + wloc.protocol + "//" + wloc.host);

    function matchOne(response, reqMethod, reqUrl) {
        var rmeth = response.method;
        var matchMethod = !rmeth || rmeth.toLowerCase() == reqMethod.toLowerCase();
        var url = response.url;
        var matchUrl = !url || url == reqUrl || (typeof url.test == "function" && url.test(reqUrl));

        return matchMethod && matchUrl;
    }

    function match(response, request) {
        var requestUrl = request.url;

        if (!/^https?:\/\//.test(requestUrl) || rCurrLoc.test(requestUrl)) {
            requestUrl = requestUrl.replace(rCurrLoc, "");
        }

        if (matchOne(response, this.getHTTPMethod(request), requestUrl)) {
            if (typeof response.response == "function") {
                var ru = response.url;
                var args = [request].concat(ru && typeof ru.exec == "function" ? ru.exec(requestUrl).slice(1) : []);
                return response.response.apply(response, args);
            }

            return true;
        }

        return false;
    }

    function makeApi(sinon) {
        sinon.fakeServer = {
            create: function () {
                var server = create(this);
                this.xhr = sinon.useFakeXMLHttpRequest();
                server.requests = [];

                this.xhr.onCreate = function (xhrObj) {
                    server.addRequest(xhrObj);
                };

                return server;
            },

            addRequest: function addRequest(xhrObj) {
                var server = this;
                push.call(this.requests, xhrObj);

                xhrObj.onSend = function () {
                    server.handleRequest(this);

                    if (server.autoRespond && !server.responding) {
                        setTimeout(function () {
                            server.responding = false;
                            server.respond();
                        }, server.autoRespondAfter || 10);

                        server.responding = true;
                    }
                };
            },

            getHTTPMethod: function getHTTPMethod(request) {
                if (this.fakeHTTPMethods && /post/i.test(request.method)) {
                    var matches = (request.requestBody || "").match(/_method=([^\b;]+)/);
                    return !!matches ? matches[1] : request.method;
                }

                return request.method;
            },

            handleRequest: function handleRequest(xhr) {
                if (xhr.async) {
                    if (!this.queue) {
                        this.queue = [];
                    }

                    push.call(this.queue, xhr);
                } else {
                    this.processRequest(xhr);
                }
            },

            log: function log(response, request) {
                var str;

                str =  "Request:\n"  + sinon.format(request)  + "\n\n";
                str += "Response:\n" + sinon.format(response) + "\n\n";

                sinon.log(str);
            },

            respondWith: function respondWith(method, url, body) {
                if (arguments.length == 1 && typeof method != "function") {
                    this.response = responseArray(method);
                    return;
                }

                if (!this.responses) { this.responses = []; }

                if (arguments.length == 1) {
                    body = method;
                    url = method = null;
                }

                if (arguments.length == 2) {
                    body = url;
                    url = method;
                    method = null;
                }

                push.call(this.responses, {
                    method: method,
                    url: url,
                    response: typeof body == "function" ? body : responseArray(body)
                });
            },

            respond: function respond() {
                if (arguments.length > 0) {
                    this.respondWith.apply(this, arguments);
                }

                var queue = this.queue || [];
                var requests = queue.splice(0, queue.length);
                var request;

                while (request = requests.shift()) {
                    this.processRequest(request);
                }
            },

            processRequest: function processRequest(request) {
                try {
                    if (request.aborted) {
                        return;
                    }

                    var response = this.response || [404, {}, ""];

                    if (this.responses) {
                        for (var l = this.responses.length, i = l - 1; i >= 0; i--) {
                            if (match.call(this, this.responses[i], request)) {
                                response = this.responses[i].response;
                                break;
                            }
                        }
                    }

                    if (request.readyState != 4) {
                        this.log(response, request);

                        request.respond(response[0], response[1], response[2]);
                    }
                } catch (e) {
                    sinon.logError("Fake server request processing", e);
                }
            },

            restore: function restore() {
                return this.xhr.restore && this.xhr.restore.apply(this.xhr, arguments);
            }
        };
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        var sinon = require("./core");
        require("./fake_xml_http_request");
        makeApi(sinon);
        module.exports = sinon;
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else {
        makeApi(sinon);
    }
}());

},{"./core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js","./fake_xml_http_request":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/fake_xml_http_request.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/fake_timers.js":[function(require,module,exports){
(function (global){
/*global lolex */

/**
 * Fake timer API
 * setTimeout
 * setInterval
 * clearTimeout
 * clearInterval
 * tick
 * reset
 * Date
 *
 * Inspired by jsUnitMockTimeOut from JsUnit
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2013 Christian Johansen
 */
"use strict";

if (typeof sinon == "undefined") {
    var sinon = {};
}

(function (global) {
    function makeApi(sinon, lol) {
        var llx = typeof lolex !== "undefined" ? lolex : lol;

        sinon.useFakeTimers = function () {
            var now, methods = Array.prototype.slice.call(arguments);

            if (typeof methods[0] === "string") {
                now = 0;
            } else {
                now = methods.shift();
            }

            var clock = llx.install(now || 0, methods);
            clock.restore = clock.uninstall;
            return clock;
        };

        sinon.clock = {
            create: function (now) {
                return llx.createClock(now);
            }
        };

        sinon.timers = {
            setTimeout: setTimeout,
            clearTimeout: clearTimeout,
            setImmediate: (typeof setImmediate !== "undefined" ? setImmediate : undefined),
            clearImmediate: (typeof clearImmediate !== "undefined" ? clearImmediate : undefined),
            setInterval: setInterval,
            clearInterval: clearInterval,
            Date: Date
        };
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, epxorts, module) {
        var sinon = require("./core");
        makeApi(sinon, require("lolex"));
        module.exports = sinon;
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else {
        makeApi(sinon);
    }
}(typeof global != "undefined" && typeof global !== "function" ? global : this));

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js","lolex":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/node_modules/lolex/src/lolex.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/fake_xml_http_request.js":[function(require,module,exports){
/**
 * @depend core.js
 * @depend ../extend.js
 * @depend event.js
 * @depend ../log_error.js
 */
/**
 * Fake XMLHttpRequest object
 *
 * @author Christian Johansen (christian@cjohansen.no)
 * @license BSD
 *
 * Copyright (c) 2010-2013 Christian Johansen
 */
"use strict";

(function (global) {

    var supportsProgress = typeof ProgressEvent !== "undefined";
    var supportsCustomEvent = typeof CustomEvent !== "undefined";
    var sinonXhr = { XMLHttpRequest: global.XMLHttpRequest };
    sinonXhr.GlobalXMLHttpRequest = global.XMLHttpRequest;
    sinonXhr.GlobalActiveXObject = global.ActiveXObject;
    sinonXhr.supportsActiveX = typeof sinonXhr.GlobalActiveXObject != "undefined";
    sinonXhr.supportsXHR = typeof sinonXhr.GlobalXMLHttpRequest != "undefined";
    sinonXhr.workingXHR = sinonXhr.supportsXHR ? sinonXhr.GlobalXMLHttpRequest : sinonXhr.supportsActiveX
                                     ? function () { return new sinonXhr.GlobalActiveXObject("MSXML2.XMLHTTP.3.0") } : false;
    sinonXhr.supportsCORS = sinonXhr.supportsXHR && "withCredentials" in (new sinonXhr.GlobalXMLHttpRequest());

    /*jsl:ignore*/
    var unsafeHeaders = {
        "Accept-Charset": true,
        "Accept-Encoding": true,
        Connection: true,
        "Content-Length": true,
        Cookie: true,
        Cookie2: true,
        "Content-Transfer-Encoding": true,
        Date: true,
        Expect: true,
        Host: true,
        "Keep-Alive": true,
        Referer: true,
        TE: true,
        Trailer: true,
        "Transfer-Encoding": true,
        Upgrade: true,
        "User-Agent": true,
        Via: true
    };
    /*jsl:end*/

    function FakeXMLHttpRequest() {
        this.readyState = FakeXMLHttpRequest.UNSENT;
        this.requestHeaders = {};
        this.requestBody = null;
        this.status = 0;
        this.statusText = "";
        this.upload = new UploadProgress();
        if (sinonXhr.supportsCORS) {
            this.withCredentials = false;
        }

        var xhr = this;
        var events = ["loadstart", "load", "abort", "loadend"];

        function addEventListener(eventName) {
            xhr.addEventListener(eventName, function (event) {
                var listener = xhr["on" + eventName];

                if (listener && typeof listener == "function") {
                    listener.call(this, event);
                }
            });
        }

        for (var i = events.length - 1; i >= 0; i--) {
            addEventListener(events[i]);
        }

        if (typeof FakeXMLHttpRequest.onCreate == "function") {
            FakeXMLHttpRequest.onCreate(this);
        }
    }

    // An upload object is created for each
    // FakeXMLHttpRequest and allows upload
    // events to be simulated using uploadProgress
    // and uploadError.
    function UploadProgress() {
        this.eventListeners = {
            progress: [],
            load: [],
            abort: [],
            error: []
        }
    }

    UploadProgress.prototype.addEventListener = function addEventListener(event, listener) {
        this.eventListeners[event].push(listener);
    };

    UploadProgress.prototype.removeEventListener = function removeEventListener(event, listener) {
        var listeners = this.eventListeners[event] || [];

        for (var i = 0, l = listeners.length; i < l; ++i) {
            if (listeners[i] == listener) {
                return listeners.splice(i, 1);
            }
        }
    };

    UploadProgress.prototype.dispatchEvent = function dispatchEvent(event) {
        var listeners = this.eventListeners[event.type] || [];

        for (var i = 0, listener; (listener = listeners[i]) != null; i++) {
            listener(event);
        }
    };

    function verifyState(xhr) {
        if (xhr.readyState !== FakeXMLHttpRequest.OPENED) {
            throw new Error("INVALID_STATE_ERR");
        }

        if (xhr.sendFlag) {
            throw new Error("INVALID_STATE_ERR");
        }
    }

    function getHeader(headers, header) {
        header = header.toLowerCase();

        for (var h in headers) {
            if (h.toLowerCase() == header) {
                return h;
            }
        }

        return null;
    }

    // filtering to enable a white-list version of Sinon FakeXhr,
    // where whitelisted requests are passed through to real XHR
    function each(collection, callback) {
        if (!collection) {
            return;
        }

        for (var i = 0, l = collection.length; i < l; i += 1) {
            callback(collection[i]);
        }
    }
    function some(collection, callback) {
        for (var index = 0; index < collection.length; index++) {
            if (callback(collection[index]) === true) {
                return true;
            }
        }
        return false;
    }
    // largest arity in XHR is 5 - XHR#open
    var apply = function (obj, method, args) {
        switch (args.length) {
        case 0: return obj[method]();
        case 1: return obj[method](args[0]);
        case 2: return obj[method](args[0], args[1]);
        case 3: return obj[method](args[0], args[1], args[2]);
        case 4: return obj[method](args[0], args[1], args[2], args[3]);
        case 5: return obj[method](args[0], args[1], args[2], args[3], args[4]);
        }
    };

    FakeXMLHttpRequest.filters = [];
    FakeXMLHttpRequest.addFilter = function addFilter(fn) {
        this.filters.push(fn)
    };
    var IE6Re = /MSIE 6/;
    FakeXMLHttpRequest.defake = function defake(fakeXhr, xhrArgs) {
        var xhr = new sinonXhr.workingXHR();
        each([
            "open",
            "setRequestHeader",
            "send",
            "abort",
            "getResponseHeader",
            "getAllResponseHeaders",
            "addEventListener",
            "overrideMimeType",
            "removeEventListener"
        ], function (method) {
            fakeXhr[method] = function () {
                return apply(xhr, method, arguments);
            };
        });

        var copyAttrs = function (args) {
            each(args, function (attr) {
                try {
                    fakeXhr[attr] = xhr[attr]
                } catch (e) {
                    if (!IE6Re.test(navigator.userAgent)) {
                        throw e;
                    }
                }
            });
        };

        var stateChange = function stateChange() {
            fakeXhr.readyState = xhr.readyState;
            if (xhr.readyState >= FakeXMLHttpRequest.HEADERS_RECEIVED) {
                copyAttrs(["status", "statusText"]);
            }
            if (xhr.readyState >= FakeXMLHttpRequest.LOADING) {
                copyAttrs(["responseText", "response"]);
            }
            if (xhr.readyState === FakeXMLHttpRequest.DONE) {
                copyAttrs(["responseXML"]);
            }
            if (fakeXhr.onreadystatechange) {
                fakeXhr.onreadystatechange.call(fakeXhr, { target: fakeXhr });
            }
        };

        if (xhr.addEventListener) {
            for (var event in fakeXhr.eventListeners) {
                if (fakeXhr.eventListeners.hasOwnProperty(event)) {
                    each(fakeXhr.eventListeners[event], function (handler) {
                        xhr.addEventListener(event, handler);
                    });
                }
            }
            xhr.addEventListener("readystatechange", stateChange);
        } else {
            xhr.onreadystatechange = stateChange;
        }
        apply(xhr, "open", xhrArgs);
    };
    FakeXMLHttpRequest.useFilters = false;

    function verifyRequestOpened(xhr) {
        if (xhr.readyState != FakeXMLHttpRequest.OPENED) {
            throw new Error("INVALID_STATE_ERR - " + xhr.readyState);
        }
    }

    function verifyRequestSent(xhr) {
        if (xhr.readyState == FakeXMLHttpRequest.DONE) {
            throw new Error("Request done");
        }
    }

    function verifyHeadersReceived(xhr) {
        if (xhr.async && xhr.readyState != FakeXMLHttpRequest.HEADERS_RECEIVED) {
            throw new Error("No headers received");
        }
    }

    function verifyResponseBodyType(body) {
        if (typeof body != "string") {
            var error = new Error("Attempted to respond to fake XMLHttpRequest with " +
                                 body + ", which is not a string.");
            error.name = "InvalidBodyException";
            throw error;
        }
    }

    FakeXMLHttpRequest.parseXML = function parseXML(text) {
        var xmlDoc;

        if (typeof DOMParser != "undefined") {
            var parser = new DOMParser();
            xmlDoc = parser.parseFromString(text, "text/xml");
        } else {
            xmlDoc = new ActiveXObject("Microsoft.XMLDOM");
            xmlDoc.async = "false";
            xmlDoc.loadXML(text);
        }

        return xmlDoc;
    };

    FakeXMLHttpRequest.statusCodes = {
        100: "Continue",
        101: "Switching Protocols",
        200: "OK",
        201: "Created",
        202: "Accepted",
        203: "Non-Authoritative Information",
        204: "No Content",
        205: "Reset Content",
        206: "Partial Content",
        207: "Multi-Status",
        300: "Multiple Choice",
        301: "Moved Permanently",
        302: "Found",
        303: "See Other",
        304: "Not Modified",
        305: "Use Proxy",
        307: "Temporary Redirect",
        400: "Bad Request",
        401: "Unauthorized",
        402: "Payment Required",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        406: "Not Acceptable",
        407: "Proxy Authentication Required",
        408: "Request Timeout",
        409: "Conflict",
        410: "Gone",
        411: "Length Required",
        412: "Precondition Failed",
        413: "Request Entity Too Large",
        414: "Request-URI Too Long",
        415: "Unsupported Media Type",
        416: "Requested Range Not Satisfiable",
        417: "Expectation Failed",
        422: "Unprocessable Entity",
        500: "Internal Server Error",
        501: "Not Implemented",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout",
        505: "HTTP Version Not Supported"
    };

    function makeApi(sinon) {
        sinon.xhr = sinonXhr;

        sinon.extend(FakeXMLHttpRequest.prototype, sinon.EventTarget, {
            async: true,

            open: function open(method, url, async, username, password) {
                this.method = method;
                this.url = url;
                this.async = typeof async == "boolean" ? async : true;
                this.username = username;
                this.password = password;
                this.responseText = null;
                this.responseXML = null;
                this.requestHeaders = {};
                this.sendFlag = false;

                if (FakeXMLHttpRequest.useFilters === true) {
                    var xhrArgs = arguments;
                    var defake = some(FakeXMLHttpRequest.filters, function (filter) {
                        return filter.apply(this, xhrArgs)
                    });
                    if (defake) {
                        return FakeXMLHttpRequest.defake(this, arguments);
                    }
                }
                this.readyStateChange(FakeXMLHttpRequest.OPENED);
            },

            readyStateChange: function readyStateChange(state) {
                this.readyState = state;

                if (typeof this.onreadystatechange == "function") {
                    try {
                        this.onreadystatechange();
                    } catch (e) {
                        sinon.logError("Fake XHR onreadystatechange handler", e);
                    }
                }

                this.dispatchEvent(new sinon.Event("readystatechange"));

                switch (this.readyState) {
                    case FakeXMLHttpRequest.DONE:
                        this.dispatchEvent(new sinon.Event("load", false, false, this));
                        this.dispatchEvent(new sinon.Event("loadend", false, false, this));
                        this.upload.dispatchEvent(new sinon.Event("load", false, false, this));
                        if (supportsProgress) {
                            this.upload.dispatchEvent(new sinon.ProgressEvent("progress", {loaded: 100, total: 100}));
                        }
                        break;
                }
            },

            setRequestHeader: function setRequestHeader(header, value) {
                verifyState(this);

                if (unsafeHeaders[header] || /^(Sec-|Proxy-)/.test(header)) {
                    throw new Error("Refused to set unsafe header \"" + header + "\"");
                }

                if (this.requestHeaders[header]) {
                    this.requestHeaders[header] += "," + value;
                } else {
                    this.requestHeaders[header] = value;
                }
            },

            // Helps testing
            setResponseHeaders: function setResponseHeaders(headers) {
                verifyRequestOpened(this);
                this.responseHeaders = {};

                for (var header in headers) {
                    if (headers.hasOwnProperty(header)) {
                        this.responseHeaders[header] = headers[header];
                    }
                }

                if (this.async) {
                    this.readyStateChange(FakeXMLHttpRequest.HEADERS_RECEIVED);
                } else {
                    this.readyState = FakeXMLHttpRequest.HEADERS_RECEIVED;
                }
            },

            // Currently treats ALL data as a DOMString (i.e. no Document)
            send: function send(data) {
                verifyState(this);

                if (!/^(get|head)$/i.test(this.method)) {
                    var contentType = getHeader(this.requestHeaders, "Content-Type");
                    if (this.requestHeaders[contentType]) {
                        var value = this.requestHeaders[contentType].split(";");
                        this.requestHeaders[contentType] = value[0] + ";charset=utf-8";
                    } else {
                        this.requestHeaders["Content-Type"] = "text/plain;charset=utf-8";
                    }

                    this.requestBody = data;
                }

                this.errorFlag = false;
                this.sendFlag = this.async;
                this.readyStateChange(FakeXMLHttpRequest.OPENED);

                if (typeof this.onSend == "function") {
                    this.onSend(this);
                }

                this.dispatchEvent(new sinon.Event("loadstart", false, false, this));
            },

            abort: function abort() {
                this.aborted = true;
                this.responseText = null;
                this.errorFlag = true;
                this.requestHeaders = {};

                if (this.readyState > FakeXMLHttpRequest.UNSENT && this.sendFlag) {
                    this.readyStateChange(FakeXMLHttpRequest.DONE);
                    this.sendFlag = false;
                }

                this.readyState = FakeXMLHttpRequest.UNSENT;

                this.dispatchEvent(new sinon.Event("abort", false, false, this));

                this.upload.dispatchEvent(new sinon.Event("abort", false, false, this));

                if (typeof this.onerror === "function") {
                    this.onerror();
                }
            },

            getResponseHeader: function getResponseHeader(header) {
                if (this.readyState < FakeXMLHttpRequest.HEADERS_RECEIVED) {
                    return null;
                }

                if (/^Set-Cookie2?$/i.test(header)) {
                    return null;
                }

                header = getHeader(this.responseHeaders, header);

                return this.responseHeaders[header] || null;
            },

            getAllResponseHeaders: function getAllResponseHeaders() {
                if (this.readyState < FakeXMLHttpRequest.HEADERS_RECEIVED) {
                    return "";
                }

                var headers = "";

                for (var header in this.responseHeaders) {
                    if (this.responseHeaders.hasOwnProperty(header) &&
                        !/^Set-Cookie2?$/i.test(header)) {
                        headers += header + ": " + this.responseHeaders[header] + "\r\n";
                    }
                }

                return headers;
            },

            setResponseBody: function setResponseBody(body) {
                verifyRequestSent(this);
                verifyHeadersReceived(this);
                verifyResponseBodyType(body);

                var chunkSize = this.chunkSize || 10;
                var index = 0;
                this.responseText = "";

                do {
                    if (this.async) {
                        this.readyStateChange(FakeXMLHttpRequest.LOADING);
                    }

                    this.responseText += body.substring(index, index + chunkSize);
                    index += chunkSize;
                } while (index < body.length);

                var type = this.getResponseHeader("Content-Type");

                if (this.responseText &&
                    (!type || /(text\/xml)|(application\/xml)|(\+xml)/.test(type))) {
                    try {
                        this.responseXML = FakeXMLHttpRequest.parseXML(this.responseText);
                    } catch (e) {
                        // Unable to parse XML - no biggie
                    }
                }

                this.readyStateChange(FakeXMLHttpRequest.DONE);
            },

            respond: function respond(status, headers, body) {
                this.status = typeof status == "number" ? status : 200;
                this.statusText = FakeXMLHttpRequest.statusCodes[this.status];
                this.setResponseHeaders(headers || {});
                this.setResponseBody(body || "");
            },

            uploadProgress: function uploadProgress(progressEventRaw) {
                if (supportsProgress) {
                    this.upload.dispatchEvent(new sinon.ProgressEvent("progress", progressEventRaw));
                }
            },

            uploadError: function uploadError(error) {
                if (supportsCustomEvent) {
                    this.upload.dispatchEvent(new sinon.CustomEvent("error", {detail: error}));
                }
            }
        });

        sinon.extend(FakeXMLHttpRequest, {
            UNSENT: 0,
            OPENED: 1,
            HEADERS_RECEIVED: 2,
            LOADING: 3,
            DONE: 4
        });

        sinon.useFakeXMLHttpRequest = function () {
            FakeXMLHttpRequest.restore = function restore(keepOnCreate) {
                if (sinonXhr.supportsXHR) {
                    global.XMLHttpRequest = sinonXhr.GlobalXMLHttpRequest;
                }

                if (sinonXhr.supportsActiveX) {
                    global.ActiveXObject = sinonXhr.GlobalActiveXObject;
                }

                delete FakeXMLHttpRequest.restore;

                if (keepOnCreate !== true) {
                    delete FakeXMLHttpRequest.onCreate;
                }
            };
            if (sinonXhr.supportsXHR) {
                global.XMLHttpRequest = FakeXMLHttpRequest;
            }

            if (sinonXhr.supportsActiveX) {
                global.ActiveXObject = function ActiveXObject(objId) {
                    if (objId == "Microsoft.XMLHTTP" || /^Msxml2\.XMLHTTP/i.test(objId)) {

                        return new FakeXMLHttpRequest();
                    }

                    return new sinonXhr.GlobalActiveXObject(objId);
                };
            }

            return FakeXMLHttpRequest;
        };

        sinon.FakeXMLHttpRequest = FakeXMLHttpRequest;
    }

    var isNode = typeof module !== "undefined" && module.exports && typeof require == "function";
    var isAMD = typeof define === "function" && typeof define.amd === "object" && define.amd;

    function loadDependencies(require, exports, module) {
        var sinon = require("./core");
        require("./event");
        makeApi(sinon);
        module.exports = sinon;
    }

    if (isAMD) {
        define(loadDependencies);
    } else if (isNode) {
        loadDependencies(require, module.exports, module);
    } else if (typeof sinon === "undefined") {
        return;
    } else {
        makeApi(sinon);
    }

})(typeof self !== "undefined" ? self : this);

},{"./core":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/core.js","./event":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/lib/sinon/util/event.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/node_modules/formatio/lib/formatio.js":[function(require,module,exports){
(function (global){
((typeof define === "function" && define.amd && function (m) {
    define("formatio", ["samsam"], m);
}) || (typeof module === "object" && function (m) {
    module.exports = m(require("samsam"));
}) || function (m) { this.formatio = m(this.samsam); }
)(function (samsam) {
    "use strict";

    var formatio = {
        excludeConstructors: ["Object", /^.$/],
        quoteStrings: true,
        limitChildrenCount: 0
    };

    var hasOwn = Object.prototype.hasOwnProperty;

    var specialObjects = [];
    if (typeof global !== "undefined") {
        specialObjects.push({ object: global, value: "[object global]" });
    }
    if (typeof document !== "undefined") {
        specialObjects.push({
            object: document,
            value: "[object HTMLDocument]"
        });
    }
    if (typeof window !== "undefined") {
        specialObjects.push({ object: window, value: "[object Window]" });
    }

    function functionName(func) {
        if (!func) { return ""; }
        if (func.displayName) { return func.displayName; }
        if (func.name) { return func.name; }
        var matches = func.toString().match(/function\s+([^\(]+)/m);
        return (matches && matches[1]) || "";
    }

    function constructorName(f, object) {
        var name = functionName(object && object.constructor);
        var excludes = f.excludeConstructors ||
                formatio.excludeConstructors || [];

        var i, l;
        for (i = 0, l = excludes.length; i < l; ++i) {
            if (typeof excludes[i] === "string" && excludes[i] === name) {
                return "";
            } else if (excludes[i].test && excludes[i].test(name)) {
                return "";
            }
        }

        return name;
    }

    function isCircular(object, objects) {
        if (typeof object !== "object") { return false; }
        var i, l;
        for (i = 0, l = objects.length; i < l; ++i) {
            if (objects[i] === object) { return true; }
        }
        return false;
    }

    function ascii(f, object, processed, indent) {
        if (typeof object === "string") {
            var qs = f.quoteStrings;
            var quote = typeof qs !== "boolean" || qs;
            return processed || quote ? '"' + object + '"' : object;
        }

        if (typeof object === "function" && !(object instanceof RegExp)) {
            return ascii.func(object);
        }

        processed = processed || [];

        if (isCircular(object, processed)) { return "[Circular]"; }

        if (Object.prototype.toString.call(object) === "[object Array]") {
            return ascii.array.call(f, object, processed);
        }

        if (!object) { return String((1/object) === -Infinity ? "-0" : object); }
        if (samsam.isElement(object)) { return ascii.element(object); }

        if (typeof object.toString === "function" &&
                object.toString !== Object.prototype.toString) {
            return object.toString();
        }

        var i, l;
        for (i = 0, l = specialObjects.length; i < l; i++) {
            if (object === specialObjects[i].object) {
                return specialObjects[i].value;
            }
        }

        return ascii.object.call(f, object, processed, indent);
    }

    ascii.func = function (func) {
        return "function " + functionName(func) + "() {}";
    };

    ascii.array = function (array, processed) {
        processed = processed || [];
        processed.push(array);
        var pieces = [];
        var i, l;
        l = (this.limitChildrenCount > 0) ? 
            Math.min(this.limitChildrenCount, array.length) : array.length;

        for (i = 0; i < l; ++i) {
            pieces.push(ascii(this, array[i], processed));
        }

        if(l < array.length)
            pieces.push("[... " + (array.length - l) + " more elements]");

        return "[" + pieces.join(", ") + "]";
    };

    ascii.object = function (object, processed, indent) {
        processed = processed || [];
        processed.push(object);
        indent = indent || 0;
        var pieces = [], properties = samsam.keys(object).sort();
        var length = 3;
        var prop, str, obj, i, k, l;
        l = (this.limitChildrenCount > 0) ? 
            Math.min(this.limitChildrenCount, properties.length) : properties.length;

        for (i = 0; i < l; ++i) {
            prop = properties[i];
            obj = object[prop];

            if (isCircular(obj, processed)) {
                str = "[Circular]";
            } else {
                str = ascii(this, obj, processed, indent + 2);
            }

            str = (/\s/.test(prop) ? '"' + prop + '"' : prop) + ": " + str;
            length += str.length;
            pieces.push(str);
        }

        var cons = constructorName(this, object);
        var prefix = cons ? "[" + cons + "] " : "";
        var is = "";
        for (i = 0, k = indent; i < k; ++i) { is += " "; }

        if(l < properties.length)
            pieces.push("[... " + (properties.length - l) + " more elements]");

        if (length + indent > 80) {
            return prefix + "{\n  " + is + pieces.join(",\n  " + is) + "\n" +
                is + "}";
        }
        return prefix + "{ " + pieces.join(", ") + " }";
    };

    ascii.element = function (element) {
        var tagName = element.tagName.toLowerCase();
        var attrs = element.attributes, attr, pairs = [], attrName, i, l, val;

        for (i = 0, l = attrs.length; i < l; ++i) {
            attr = attrs.item(i);
            attrName = attr.nodeName.toLowerCase().replace("html:", "");
            val = attr.nodeValue;
            if (attrName !== "contenteditable" || val !== "inherit") {
                if (!!val) { pairs.push(attrName + "=\"" + val + "\""); }
            }
        }

        var formatted = "<" + tagName + (pairs.length > 0 ? " " : "");
        var content = element.innerHTML;

        if (content.length > 20) {
            content = content.substr(0, 20) + "[...]";
        }

        var res = formatted + pairs.join(" ") + ">" + content +
                "</" + tagName + ">";

        return res.replace(/ contentEditable="inherit"/, "");
    };

    function Formatio(options) {
        for (var opt in options) {
            this[opt] = options[opt];
        }
    }

    Formatio.prototype = {
        functionName: functionName,

        configure: function (options) {
            return new Formatio(options);
        },

        constructorName: function (object) {
            return constructorName(this, object);
        },

        ascii: function (object, processed, indent) {
            return ascii(this, object, processed, indent);
        }
    };

    return Formatio.prototype;
});

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"samsam":"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/node_modules/formatio/node_modules/samsam/lib/samsam.js"}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/node_modules/formatio/node_modules/samsam/lib/samsam.js":[function(require,module,exports){
((typeof define === "function" && define.amd && function (m) { define("samsam", m); }) ||
 (typeof module === "object" &&
      function (m) { module.exports = m(); }) || // Node
 function (m) { this.samsam = m(); } // Browser globals
)(function () {
    var o = Object.prototype;
    var div = typeof document !== "undefined" && document.createElement("div");

    function isNaN(value) {
        // Unlike global isNaN, this avoids type coercion
        // typeof check avoids IE host object issues, hat tip to
        // lodash
        var val = value; // JsLint thinks value !== value is "weird"
        return typeof value === "number" && value !== val;
    }

    function getClass(value) {
        // Returns the internal [[Class]] by calling Object.prototype.toString
        // with the provided value as this. Return value is a string, naming the
        // internal class, e.g. "Array"
        return o.toString.call(value).split(/[ \]]/)[1];
    }

    /**
     * @name samsam.isArguments
     * @param Object object
     *
     * Returns ``true`` if ``object`` is an ``arguments`` object,
     * ``false`` otherwise.
     */
    function isArguments(object) {
        if (getClass(object) === 'Arguments') { return true; }
        if (typeof object !== "object" || typeof object.length !== "number" ||
                getClass(object) === "Array") {
            return false;
        }
        if (typeof object.callee == "function") { return true; }
        try {
            object[object.length] = 6;
            delete object[object.length];
        } catch (e) {
            return true;
        }
        return false;
    }

    /**
     * @name samsam.isElement
     * @param Object object
     *
     * Returns ``true`` if ``object`` is a DOM element node. Unlike
     * Underscore.js/lodash, this function will return ``false`` if ``object``
     * is an *element-like* object, i.e. a regular object with a ``nodeType``
     * property that holds the value ``1``.
     */
    function isElement(object) {
        if (!object || object.nodeType !== 1 || !div) { return false; }
        try {
            object.appendChild(div);
            object.removeChild(div);
        } catch (e) {
            return false;
        }
        return true;
    }

    /**
     * @name samsam.keys
     * @param Object object
     *
     * Return an array of own property names.
     */
    function keys(object) {
        var ks = [], prop;
        for (prop in object) {
            if (o.hasOwnProperty.call(object, prop)) { ks.push(prop); }
        }
        return ks;
    }

    /**
     * @name samsam.isDate
     * @param Object value
     *
     * Returns true if the object is a ``Date``, or *date-like*. Duck typing
     * of date objects work by checking that the object has a ``getTime``
     * function whose return value equals the return value from the object's
     * ``valueOf``.
     */
    function isDate(value) {
        return typeof value.getTime == "function" &&
            value.getTime() == value.valueOf();
    }

    /**
     * @name samsam.isNegZero
     * @param Object value
     *
     * Returns ``true`` if ``value`` is ``-0``.
     */
    function isNegZero(value) {
        return value === 0 && 1 / value === -Infinity;
    }

    /**
     * @name samsam.equal
     * @param Object obj1
     * @param Object obj2
     *
     * Returns ``true`` if two objects are strictly equal. Compared to
     * ``===`` there are two exceptions:
     *
     *   - NaN is considered equal to NaN
     *   - -0 and +0 are not considered equal
     */
    function identical(obj1, obj2) {
        if (obj1 === obj2 || (isNaN(obj1) && isNaN(obj2))) {
            return obj1 !== 0 || isNegZero(obj1) === isNegZero(obj2);
        }
    }


    /**
     * @name samsam.deepEqual
     * @param Object obj1
     * @param Object obj2
     *
     * Deep equal comparison. Two values are "deep equal" if:
     *
     *   - They are equal, according to samsam.identical
     *   - They are both date objects representing the same time
     *   - They are both arrays containing elements that are all deepEqual
     *   - They are objects with the same set of properties, and each property
     *     in ``obj1`` is deepEqual to the corresponding property in ``obj2``
     *
     * Supports cyclic objects.
     */
    function deepEqualCyclic(obj1, obj2) {

        // used for cyclic comparison
        // contain already visited objects
        var objects1 = [],
            objects2 = [],
        // contain pathes (position in the object structure)
        // of the already visited objects
        // indexes same as in objects arrays
            paths1 = [],
            paths2 = [],
        // contains combinations of already compared objects
        // in the manner: { "$1['ref']$2['ref']": true }
            compared = {};

        /**
         * used to check, if the value of a property is an object
         * (cyclic logic is only needed for objects)
         * only needed for cyclic logic
         */
        function isObject(value) {

            if (typeof value === 'object' && value !== null &&
                    !(value instanceof Boolean) &&
                    !(value instanceof Date)    &&
                    !(value instanceof Number)  &&
                    !(value instanceof RegExp)  &&
                    !(value instanceof String)) {

                return true;
            }

            return false;
        }

        /**
         * returns the index of the given object in the
         * given objects array, -1 if not contained
         * only needed for cyclic logic
         */
        function getIndex(objects, obj) {

            var i;
            for (i = 0; i < objects.length; i++) {
                if (objects[i] === obj) {
                    return i;
                }
            }

            return -1;
        }

        // does the recursion for the deep equal check
        return (function deepEqual(obj1, obj2, path1, path2) {
            var type1 = typeof obj1;
            var type2 = typeof obj2;

            // == null also matches undefined
            if (obj1 === obj2 ||
                    isNaN(obj1) || isNaN(obj2) ||
                    obj1 == null || obj2 == null ||
                    type1 !== "object" || type2 !== "object") {

                return identical(obj1, obj2);
            }

            // Elements are only equal if identical(expected, actual)
            if (isElement(obj1) || isElement(obj2)) { return false; }

            var isDate1 = isDate(obj1), isDate2 = isDate(obj2);
            if (isDate1 || isDate2) {
                if (!isDate1 || !isDate2 || obj1.getTime() !== obj2.getTime()) {
                    return false;
                }
            }

            if (obj1 instanceof RegExp && obj2 instanceof RegExp) {
                if (obj1.toString() !== obj2.toString()) { return false; }
            }

            var class1 = getClass(obj1);
            var class2 = getClass(obj2);
            var keys1 = keys(obj1);
            var keys2 = keys(obj2);

            if (isArguments(obj1) || isArguments(obj2)) {
                if (obj1.length !== obj2.length) { return false; }
            } else {
                if (type1 !== type2 || class1 !== class2 ||
                        keys1.length !== keys2.length) {
                    return false;
                }
            }

            var key, i, l,
                // following vars are used for the cyclic logic
                value1, value2,
                isObject1, isObject2,
                index1, index2,
                newPath1, newPath2;

            for (i = 0, l = keys1.length; i < l; i++) {
                key = keys1[i];
                if (!o.hasOwnProperty.call(obj2, key)) {
                    return false;
                }

                // Start of the cyclic logic

                value1 = obj1[key];
                value2 = obj2[key];

                isObject1 = isObject(value1);
                isObject2 = isObject(value2);

                // determine, if the objects were already visited
                // (it's faster to check for isObject first, than to
                // get -1 from getIndex for non objects)
                index1 = isObject1 ? getIndex(objects1, value1) : -1;
                index2 = isObject2 ? getIndex(objects2, value2) : -1;

                // determine the new pathes of the objects
                // - for non cyclic objects the current path will be extended
                //   by current property name
                // - for cyclic objects the stored path is taken
                newPath1 = index1 !== -1
                    ? paths1[index1]
                    : path1 + '[' + JSON.stringify(key) + ']';
                newPath2 = index2 !== -1
                    ? paths2[index2]
                    : path2 + '[' + JSON.stringify(key) + ']';

                // stop recursion if current objects are already compared
                if (compared[newPath1 + newPath2]) {
                    return true;
                }

                // remember the current objects and their pathes
                if (index1 === -1 && isObject1) {
                    objects1.push(value1);
                    paths1.push(newPath1);
                }
                if (index2 === -1 && isObject2) {
                    objects2.push(value2);
                    paths2.push(newPath2);
                }

                // remember that the current objects are already compared
                if (isObject1 && isObject2) {
                    compared[newPath1 + newPath2] = true;
                }

                // End of cyclic logic

                // neither value1 nor value2 is a cycle
                // continue with next level
                if (!deepEqual(value1, value2, newPath1, newPath2)) {
                    return false;
                }
            }

            return true;

        }(obj1, obj2, '$1', '$2'));
    }

    var match;

    function arrayContains(array, subset) {
        if (subset.length === 0) { return true; }
        var i, l, j, k;
        for (i = 0, l = array.length; i < l; ++i) {
            if (match(array[i], subset[0])) {
                for (j = 0, k = subset.length; j < k; ++j) {
                    if (!match(array[i + j], subset[j])) { return false; }
                }
                return true;
            }
        }
        return false;
    }

    /**
     * @name samsam.match
     * @param Object object
     * @param Object matcher
     *
     * Compare arbitrary value ``object`` with matcher.
     */
    match = function match(object, matcher) {
        if (matcher && typeof matcher.test === "function") {
            return matcher.test(object);
        }

        if (typeof matcher === "function") {
            return matcher(object) === true;
        }

        if (typeof matcher === "string") {
            matcher = matcher.toLowerCase();
            var notNull = typeof object === "string" || !!object;
            return notNull &&
                (String(object)).toLowerCase().indexOf(matcher) >= 0;
        }

        if (typeof matcher === "number") {
            return matcher === object;
        }

        if (typeof matcher === "boolean") {
            return matcher === object;
        }

        if (typeof(matcher) === "undefined") {
            return typeof(object) === "undefined";
        }

        if (matcher === null) {
            return object === null;
        }

        if (getClass(object) === "Array" && getClass(matcher) === "Array") {
            return arrayContains(object, matcher);
        }

        if (matcher && typeof matcher === "object") {
            if (matcher === object) {
                return true;
            }
            var prop;
            for (prop in matcher) {
                var value = object[prop];
                if (typeof value === "undefined" &&
                        typeof object.getAttribute === "function") {
                    value = object.getAttribute(prop);
                }
                if (matcher[prop] === null || typeof matcher[prop] === 'undefined') {
                    if (value !== matcher[prop]) {
                        return false;
                    }
                } else if (typeof  value === "undefined" || !match(value, matcher[prop])) {
                    return false;
                }
            }
            return true;
        }

        throw new Error("Matcher was not a string, a number, a " +
                        "function, a boolean or an object");
    };

    return {
        isArguments: isArguments,
        isElement: isElement,
        isDate: isDate,
        isNegZero: isNegZero,
        identical: identical,
        deepEqual: deepEqualCyclic,
        match: match,
        keys: keys
    };
});

},{}],"/Users/stanley/code/rsvp.js/node_modules/promises-aplus-tests-phantom/node_modules/sinon/node_modules/lolex/src/lolex.js":[function(require,module,exports){
(function (global){
/*jslint eqeqeq: false, plusplus: false, evil: true, onevar: false, browser: true, forin: false*/
/*global global*/
/**
 * @author Christian Johansen (christian@cjohansen.no) and contributors
 * @license BSD
 *
 * Copyright (c) 2010-2014 Christian Johansen
 */
"use strict";

// node expects setTimeout/setInterval to return a fn object w/ .ref()/.unref()
// browsers, a number.
// see https://github.com/cjohansen/Sinon.JS/pull/436
var timeoutResult = setTimeout(function() {}, 0);
var addTimerReturnsObject = typeof timeoutResult === "object";
clearTimeout(timeoutResult);

var NativeDate = Date;
var id = 1;

/**
 * Parse strings like "01:10:00" (meaning 1 hour, 10 minutes, 0 seconds) into
 * number of milliseconds. This is used to support human-readable strings passed
 * to clock.tick()
 */
function parseTime(str) {
    if (!str) {
        return 0;
    }

    var strings = str.split(":");
    var l = strings.length, i = l;
    var ms = 0, parsed;

    if (l > 3 || !/^(\d\d:){0,2}\d\d?$/.test(str)) {
        throw new Error("tick only understands numbers and 'h:m:s'");
    }

    while (i--) {
        parsed = parseInt(strings[i], 10);

        if (parsed >= 60) {
            throw new Error("Invalid time " + str);
        }

        ms += parsed * Math.pow(60, (l - i - 1));
    }

    return ms * 1000;
}

/**
 * Used to grok the `now` parameter to createClock.
 */
function getEpoch(epoch) {
    if (!epoch) { return 0; }
    if (typeof epoch.getTime === "function") { return epoch.getTime(); }
    if (typeof epoch === "number") { return epoch; }
    throw new TypeError("now should be milliseconds since UNIX epoch");
}

function inRange(from, to, timer) {
    return timer && timer.callAt >= from && timer.callAt <= to;
}

function mirrorDateProperties(target, source) {
    if (source.now) {
        target.now = function now() {
            return target.clock.now;
        };
    } else {
        delete target.now;
    }

    if (source.toSource) {
        target.toSource = function toSource() {
            return source.toSource();
        };
    } else {
        delete target.toSource;
    }

    target.toString = function toString() {
        return source.toString();
    };

    target.prototype = source.prototype;
    target.parse = source.parse;
    target.UTC = source.UTC;
    target.prototype.toUTCString = source.prototype.toUTCString;

    for (var prop in source) {
        if (source.hasOwnProperty(prop)) {
            target[prop] = source[prop];
        }
    }

    return target;
}

function createDate() {
    function ClockDate(year, month, date, hour, minute, second, ms) {
        // Defensive and verbose to avoid potential harm in passing
        // explicit undefined when user does not pass argument
        switch (arguments.length) {
        case 0:
            return new NativeDate(ClockDate.clock.now);
        case 1:
            return new NativeDate(year);
        case 2:
            return new NativeDate(year, month);
        case 3:
            return new NativeDate(year, month, date);
        case 4:
            return new NativeDate(year, month, date, hour);
        case 5:
            return new NativeDate(year, month, date, hour, minute);
        case 6:
            return new NativeDate(year, month, date, hour, minute, second);
        default:
            return new NativeDate(year, month, date, hour, minute, second, ms);
        }
    }

    return mirrorDateProperties(ClockDate, NativeDate);
}

function addTimer(clock, timer) {
    if (typeof timer.func === "undefined") {
        throw new Error("Callback must be provided to timer calls");
    }

    if (!clock.timers) {
        clock.timers = {};
    }

    timer.id = id++;
    timer.createdAt = clock.now;
    timer.callAt = clock.now + (timer.delay || 0);

    clock.timers[timer.id] = timer;

    if (addTimerReturnsObject) {
        return {
            id: timer.id,
            ref: function() {},
            unref: function() {}
        };
    }
    else {
        return timer.id;
    }
}

function firstTimerInRange(clock, from, to) {
    var timers = clock.timers, timer = null;

    for (var id in timers) {
        if (!inRange(from, to, timers[id])) {
            continue;
        }

        if (!timer || ~compareTimers(timer, timers[id])) {
            timer = timers[id];
        }
    }

    return timer;
}

function compareTimers(a, b) {
    // Sort first by absolute timing
    if (a.callAt < b.callAt) {
        return -1;
    }
    if (a.callAt > b.callAt) {
        return 1;
    }

    // Sort next by immediate, immediate timers take precedence
    if (a.immediate && !b.immediate) {
        return -1;
    }
    if (!a.immediate && b.immediate) {
        return 1;
    }

    // Sort next by creation time, earlier-created timers take precedence
    if (a.createdAt < b.createdAt) {
        return -1;
    }
    if (a.createdAt > b.createdAt) {
        return 1;
    }

    // Sort next by id, lower-id timers take precedence
    if (a.id < b.id) {
        return -1;
    }
    if (a.id > b.id) {
        return 1;
    }

    // As timer ids are unique, no fallback `0` is necessary
}

function callTimer(clock, timer) {
    if (typeof timer.interval == "number") {
        clock.timers[timer.id].callAt += timer.interval;
    } else {
        delete clock.timers[timer.id];
    }

    try {
        if (typeof timer.func == "function") {
            timer.func.apply(null, timer.args);
        } else {
            eval(timer.func);
        }
    } catch (e) {
        var exception = e;
    }

    if (!clock.timers[timer.id]) {
        if (exception) {
            throw exception;
        }
        return;
    }

    if (exception) {
        throw exception;
    }
}

function uninstall(clock, target) {
    var method;

    for (var i = 0, l = clock.methods.length; i < l; i++) {
        method = clock.methods[i];

        if (target[method].hadOwnProperty) {
            target[method] = clock["_" + method];
        } else {
            try {
                delete target[method];
            } catch (e) {}
        }
    }

    // Prevent multiple executions which will completely remove these props
    clock.methods = [];
}

function hijackMethod(target, method, clock) {
    clock[method].hadOwnProperty = Object.prototype.hasOwnProperty.call(target, method);
    clock["_" + method] = target[method];

    if (method == "Date") {
        var date = mirrorDateProperties(clock[method], target[method]);
        target[method] = date;
    } else {
        target[method] = function () {
            return clock[method].apply(clock, arguments);
        };

        for (var prop in clock[method]) {
            if (clock[method].hasOwnProperty(prop)) {
                target[method][prop] = clock[method][prop];
            }
        }
    }

    target[method].clock = clock;
}

var timers = {
    setTimeout: setTimeout,
    clearTimeout: clearTimeout,
    setImmediate: (typeof setImmediate !== "undefined" ? setImmediate : undefined),
    clearImmediate: (typeof clearImmediate !== "undefined" ? clearImmediate: undefined),
    setInterval: setInterval,
    clearInterval: clearInterval,
    Date: Date
};

var keys = Object.keys || function (obj) {
    var ks = [];
    for (var key in obj) {
        ks.push(key);
    }
    return ks;
};

exports.timers = timers;

var createClock = exports.createClock = function (now) {
    var clock = {
        now: getEpoch(now),
        timeouts: {},
        Date: createDate()
    };

    clock.Date.clock = clock;

    clock.setTimeout = function setTimeout(func, timeout) {
        return addTimer(clock, {
            func: func,
            args: Array.prototype.slice.call(arguments, 2),
            delay: timeout
        });
    };

    clock.clearTimeout = function clearTimeout(timerId) {
        if (!timerId) {
            // null appears to be allowed in most browsers, and appears to be
            // relied upon by some libraries, like Bootstrap carousel
            return;
        }
        if (!clock.timers) {
            clock.timers = [];
        }
        // in Node, timerId is an object with .ref()/.unref(), and
        // its .id field is the actual timer id.
        if (typeof timerId === "object") {
            timerId = timerId.id
        }
        if (timerId in clock.timers) {
            delete clock.timers[timerId];
        }
    };

    clock.setInterval = function setInterval(func, timeout) {
        return addTimer(clock, {
            func: func,
            args: Array.prototype.slice.call(arguments, 2),
            delay: timeout,
            interval: timeout
        });
    };

    clock.clearInterval = function clearInterval(timerId) {
        clock.clearTimeout(timerId);
    };

    clock.setImmediate = function setImmediate(func) {
        return addTimer(clock, {
            func: func,
            args: Array.prototype.slice.call(arguments, 1),
            immediate: true
        });
    };

    clock.clearImmediate = function clearImmediate(timerId) {
        clock.clearTimeout(timerId);
    };

    clock.tick = function tick(ms) {
        ms = typeof ms == "number" ? ms : parseTime(ms);
        var tickFrom = clock.now, tickTo = clock.now + ms, previous = clock.now;
        var timer = firstTimerInRange(clock, tickFrom, tickTo);

        var firstException;
        while (timer && tickFrom <= tickTo) {
            if (clock.timers[timer.id]) {
                tickFrom = clock.now = timer.callAt;
                try {
                    callTimer(clock, timer);
                } catch (e) {
                    firstException = firstException || e;
                }
            }

            timer = firstTimerInRange(clock, previous, tickTo);
            previous = tickFrom;
        }

        clock.now = tickTo;

        if (firstException) {
            throw firstException;
        }

        return clock.now;
    };

    clock.reset = function reset() {
        clock.timers = {};
    };

    return clock;
};

exports.install = function install(target, now, toFake) {
    if (typeof target === "number") {
        toFake = now;
        now = target;
        target = null;
    }

    if (!target) {
        target = global;
    }

    var clock = createClock(now);

    clock.uninstall = function () {
        uninstall(clock, target);
    };

    clock.methods = toFake || [];

    if (clock.methods.length === 0) {
        clock.methods = keys(timers);
    }

    for (var i = 0, l = clock.methods.length; i < l; i++) {
        hijackMethod(target, clock.methods[i], clock);
    }

    return clock;
};

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}],"/Users/stanley/code/rsvp.js/tmp/tree_merger-tmp_dest_dir-l9APRzgC.tmp/test/extension-test.js":[function(require,module,exports){
(function (process,global){
/*global describe, specify, it, assert */

if (typeof Object.getPrototypeOf !== "function") {
  Object.getPrototypeOf = "".__proto__ === String.prototype
    ? function (object) {
      return object.__proto__;
    }
    : function (object) {
      // May break if the constructor has been tampered with
      return object.constructor.prototype;
    };
}

function keysOf(object) {
  var results = [];

  for (var key in object) {
    if (object.hasOwnProperty(key)) {
      results.push(key);
    }
  }

  return results;
}

var g = typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this;
var RSVP = g.adapter.RSVP;
var assert = require('assert');

var o_create = Object.create || function(o, props) {
  function F() {}
  F.prototype = o;

  if (typeof(props) === "object") {
    for (var prop in props) {
      if (props.hasOwnProperty((prop))) {
        F[prop] = props[prop];
      }
    }
  }
  return new F();
};

function objectEquals(obj1, obj2) {
  for (var i in obj1) {
    if (obj1.hasOwnProperty(i)) {
      if (!obj2.hasOwnProperty(i)) return false;
      if (obj1[i] != obj2[i]) return false;
    }
  }
  for (var i in obj2) {
    if (obj2.hasOwnProperty(i)) {
      if (!obj1.hasOwnProperty(i)) return false;
      if (obj1[i] != obj2[i]) return false;
    }
  }
  return true;
}

describe("RSVP extensions", function() {
  describe("self fulfillment", function(){
    it("treats self fulfillment as the recursive base case", function(done){
      var aDefer = new RSVP.defer(),
      bDefer = new RSVP.defer(),
      promiseA = aDefer.promise,
      promiseB = bDefer.promise;

      promiseA.then(function(a){
        setTimeout(function(){
          bDefer.resolve(promiseB);
        }, 1);

        return promiseB;
      });

      promiseB.then(function(c){
        done();
      })

      aDefer.resolve(promiseA);
    });
  });

  describe("Promise constructor", function() {
    it('should exist and have length 2', function() {
      assert(RSVP.Promise);
      assert.equal(RSVP.Promise.length, 2);
    });

    it('should fulfill if `resolve` is called with a value', function(done) {
      var promise = new RSVP.Promise(function(resolve) { resolve('value'); });

      promise.then(function(value) {
        assert.equal(value, 'value');
        done();
      });
    });

    it('should reject if `reject` is called with a reason', function(done) {
      var promise = new RSVP.Promise(function(resolve, reject) { reject('reason'); });

      promise.then(function() {
        assert(false);
        done();
      }, function(reason) {
        assert.equal(reason, 'reason');
        done();
      });
    });

    it('should be a constructor', function() {
      var promise = new RSVP.Promise(function() {});

      assert.equal(Object.getPrototypeOf(promise), RSVP.Promise.prototype, '[[Prototype]] equals Promise.prototype');
      assert.equal(promise.constructor, RSVP.Promise, 'constructor property of instances is set correctly');
      assert.equal(RSVP.Promise.prototype.constructor, RSVP.Promise, 'constructor property of prototype is set correctly');
    });

    it('should NOT work without `new`', function() {
      assert.throws(function(){
        RSVP.Promise(function(resolve) { resolve('value'); });
      }, TypeError)
    });

    it('should throw a `TypeError` if not given a function', function() {
      assert.throws(function () {
        new RSVP.Promise();
      }, TypeError);

      assert.throws(function () {
        new RSVP.Promise({});
      }, TypeError);

      assert.throws(function () {
        new RSVP.Promise('boo!');
      }, TypeError);
    });

    it('should reject on resolver exception', function(done) {
     new RSVP.Promise(function() {
        throw 'error';
      }).then(null, function(e) {
        assert.equal(e, 'error');
        done();
      });
    });

    it('should not resolve multiple times', function(done) {
      var resolver, rejector, fulfilled = 0, rejected = 0;
      var thenable = {
        then: function(resolve, reject) {
          resolver = resolve;
          rejector = reject;
        }
      };

      var promise = new RSVP.Promise(function(resolve) {
        resolve(1);
      });

      promise.then(function(value){
        return thenable;
      }).then(function(value){
        fulfilled++;
      }, function(reason) {
        rejected++;
      });

      setTimeout(function() {
        resolver(1);
        resolver(1);
        rejector(1);
        rejector(1);

        setTimeout(function() {
          assert.equal(fulfilled, 1);
          assert.equal(rejected, 0);
          done();
        }, 20);
      }, 20);

    });

    it('should not resolve multiple times even if not settled', function() {
      var promise = new RSVP.Promise(function(resolve) {
        resolve(new RSVP.Promise(function(resolve) {
          setTimeout(function(){
            resolve(1);
          }, 10);
        }));

        resolve(RSVP.Promise.resolve(2));
      });

      return promise.then(function(val) {
        assert.equal(val, 1);
      });
    });

    it('should not reject if already resolved (and not settled)', function() {
      var promise = new RSVP.Promise(function(resolve, reject) {
        resolve(new RSVP.Promise(function(resolve) {
          setTimeout(function(){
            resolve(1);
          }, 10);
        }));

        reject(new Error());
      });

      return promise.then(function(val) {
        assert.equal(val, 1);
      }, function(){
        assert(false, 'should not have rejected');
      });
    });

    describe('assimilation', function() {
      it('should assimilate if `resolve` is called with a fulfilled promise', function(done) {
        var originalPromise = new RSVP.Promise(function(resolve) { resolve('original value'); });
        var promise = new RSVP.Promise(function(resolve) { resolve(originalPromise); });

        promise.then(function(value) {
          assert.equal(value, 'original value');
          done();
        });
      });

      it('should assimilate if `resolve` is called with a rejected promise', function(done) {
        var originalPromise = new RSVP.Promise(function(resolve, reject) { reject('original reason'); });
        var promise = new RSVP.Promise(function(resolve) { resolve(originalPromise); });

        promise.then(function() {
          assert(false);
          done();
        }, function(reason) {
          assert.equal(reason, 'original reason');
          done();
        });
      });

      it('should assimilate if `resolve` is called with a fulfilled thenable', function(done) {
        var originalThenable = {
          then: function (onFulfilled) {
            setTimeout(function() { onFulfilled('original value'); }, 0);
          }
        };
        var promise = new RSVP.Promise(function(resolve) { resolve(originalThenable); });

        promise.then(function(value) {
          assert.equal(value, 'original value');
          done();
        });
      });

      it('should assimilate if `resolve` is called with a rejected thenable', function(done) {
        var originalThenable = {
          then: function (onFulfilled, onRejected) {
            setTimeout(function() { onRejected('original reason'); }, 0);
          }
        };
        var promise = new RSVP.Promise(function(resolve) { resolve(originalThenable); });

        promise.then(function() {
          assert(false);
          done();
        }, function(reason) {
          assert.equal(reason, 'original reason');
          done();
        });
      });


      it('should assimilate two levels deep, for fulfillment of self fulfilling promises', function(done) {
        var originalPromise, promise;
        originalPromise = new RSVP.Promise(function(resolve) {
          setTimeout(function() {
            resolve(originalPromise);
          }, 0)
        });

        promise = new RSVP.Promise(function(resolve) {
          setTimeout(function() {
            resolve(originalPromise);
          }, 0);
        });

        promise.then(function(value) {
          assert.equal(value, originalPromise);
          done();
        });
      });

      it('should assimilate two levels deep, for fulfillment', function(done) {
        var originalPromise = new RSVP.Promise(function(resolve) { resolve('original value'); });
        var nextPromise = new RSVP.Promise(function(resolve) { resolve(originalPromise); });
        var promise = new RSVP.Promise(function(resolve) { resolve(nextPromise); });

        promise.then(function(value) {
          assert.equal(value, 'original value');
          done();
        });
      });

      it('should assimilate two levels deep, for rejection', function(done) {
        var originalPromise = new RSVP.Promise(function(resolve, reject) { reject('original reason'); });
        var nextPromise = new RSVP.Promise(function(resolve) { resolve(originalPromise); });
        var promise = new RSVP.Promise(function(resolve) { resolve(nextPromise); });

        promise.then(function() {
          assert(false);
          done();
        }, function(reason) {
          assert.equal(reason, 'original reason');
          done();
        });
      });

      it('should assimilate three levels deep, mixing thenables and promises (fulfilled case)', function(done) {
        var originalPromise = new RSVP.Promise(function(resolve) { resolve('original value'); });
        var intermediateThenable = {
          then: function (onFulfilled) {
            setTimeout(function() { onFulfilled(originalPromise); }, 0);
          }
        };
        var promise = new RSVP.Promise(function(resolve) { resolve(intermediateThenable); });

        promise.then(function(value) {
          assert.equal(value, 'original value');
          done();
        });
      });

      it('should assimilate three levels deep, mixing thenables and promises (rejected case)', function(done) {
        var originalPromise = new RSVP.Promise(function(resolve, reject) { reject('original reason'); });
        var intermediateThenable = {
          then: function (onFulfilled) {
            setTimeout(function() { onFulfilled(originalPromise); }, 0);
          }
        };
        var promise = new RSVP.Promise(function(resolve) { resolve(intermediateThenable); });

        promise.then(function() {
          assert(false);
          done();
        }, function(reason) {
          assert.equal(reason, 'original reason');
          done();
        });
      });
    });
  });

  describe("RSVP.defer", function() {
    specify("It should return a resolver and promise together", function(done) {
      var deferred = RSVP.defer(), value = {};

      // resolve first to confirm that the semantics are async
      deferred.resolve(value);

      deferred.promise.then(function(passedValue) {
        assert(passedValue === value);
        done();
      });
    });

    specify("The provided resolver should support rejection", function(done) {
      var deferred = RSVP.defer(), reason = {};

      // resolve first to confirm that the semantics are async
      deferred.reject(reason);

      deferred.promise.then(null, function(passedReason) {
        assert(passedReason === reason);
        done();
      });
    });
  });

  describe("RSVP.denodeify", function() {
    specify('it should exist', function() {
      assert(RSVP.denodeify);
    });

    specify('calls node function with any arguments passed', function(done) {
      var args = null;

      function nodeFunc(arg1, arg2, arg3, cb) {
        args = [arg1, arg2, arg3];
        cb();
      }

      var denodeifiedFunc = RSVP.denodeify(nodeFunc);

      denodeifiedFunc(1, 2, 3).then(function() {
        assert(objectEquals(args, [1, 2, 3]));
        done();
      });
    });

    specify('waits for promise/thenable arguments to settle before passing them to the node function', function(done) {
      var args = null;

      function nodeFunc(arg1, arg2, arg3, cb) {
        args = [arg1, arg2, arg3];
        cb();
      }

      var denodeifiedFunc = RSVP.denodeify(nodeFunc);

      var promise = new RSVP.Promise(function(resolve) { resolve(1); });
      var thenable = { then: function (onFulfilled) { onFulfilled(2); } };
      var nonPromise = 3;
      denodeifiedFunc(promise, thenable, nonPromise).then(function() {
        assert(objectEquals(args, [1, 2, 3]));
        done();
      });
    });

    specify('fulfilled with value if node function calls back with a single argument', function(done) {
      function nodeFunc(cb) {
        cb(null, 'nodeFuncResult');
      }

      var denodeifiedFunc = RSVP.denodeify(nodeFunc);

      denodeifiedFunc().then(function(value) {
        assert.equal(value, 'nodeFuncResult');
        done();
      });
    });

    specify('fulfilled with array if if user called RSVP.denodeify(nodeFunc, true)', function(done) {
      function nodeFunc(cb) {
        cb(null, 1, 2, 3);
      }

      var denodeifiedFunc = RSVP.denodeify(nodeFunc, true);

      denodeifiedFunc().then(function(value) {
        assert(objectEquals(value, [1, 2, 3]));
        done();
      });
    });

    specify('fulfilled with hash if if user called RSVP.denodeify with an array as second argument', function(done) {
      function nodeFunc(cb) {
        cb(null, 1, 2, 3);
      }

      var denodeifiedFunc = RSVP.denodeify(nodeFunc, ['a', 'b', 'c']);

      denodeifiedFunc().then(function(value) {
        assert.equal(value.a, '1');
        assert.equal(value.b, '2');
        assert.equal(value.c, '3');
        done();
      });
    });

    specify('fulfilled with just the first value if node function returns an array', function(done) {
      function nodeFunc(cb) {
        cb(null, 42, 2, 3);
      }

      var denodeifiedFunc = RSVP.denodeify(nodeFunc);

      denodeifiedFunc().then(function(value) {
        assert.equal(value, 42);
        done();
      });
    });

    specify('rejected if node function calls back with error', function(done) {
      function nodeFunc(cb) {
        cb('bad!');
      }

      var denodeifiedFunc = RSVP.denodeify(nodeFunc);

      denodeifiedFunc().then(function() {
        assert(false);
        done();
      }, function(reason) {
        assert.equal(reason, 'bad!');
        done();
      });
    });

    specify('rejected if node function throws an exception synchronously', function(done) {
      function nodeFunc(cb) {
        throw 'bad!';
      }

      var denodeifiedFunc = RSVP.denodeify(nodeFunc);

      denodeifiedFunc().then(function() {
        assert(false);
        done();
      }, function(reason) {
        assert.equal(reason, 'bad!');
        done();
      });
    });

    if (Object.prototype.hasOwnProperty('__proto__')) {
      specify('should inherit from node function', function() {
        function nodeFunc(cb) {
          cb(undefined, 'hello');
        }
        nodeFunc.something = 'test123';

        var denodeifiedFunc = RSVP.denodeify(nodeFunc);

        assert.equal(denodeifiedFunc.something, 'test123');

        denodeifiedFunc().then(function(result) {
          assert.equal(result, 'hello');
          assert.equal(denodeifiedFunc.something, 'test123');
        }, function() {
          assert.equal(false);
        });
      });
    }

    specify('integration test showing how awesome this can be', function(done) {
      function readFile(fileName, cb) {
        setTimeout(function() {
          cb(null, 'contents of ' + fileName);
        }, 0);
      }

      var writtenTo = null;
      function writeFile(fileName, text, cb) {
        setTimeout(function () {
          writtenTo = [fileName, text];
          cb();
        }, 0);
      }

      var denodeifiedReadFile = RSVP.denodeify(readFile);
      var denodeifiedWriteFile = RSVP.denodeify(writeFile);

      denodeifiedWriteFile('dest.txt', denodeifiedReadFile('src.txt')).then(function () {
        assert(objectEquals(writtenTo, ['dest.txt', 'contents of src.txt']));
        done();
      });
    });


    specify('integration test showing how awesome this can be (with promise subclass)', function(done) {
      function readFile(fileName, cb) {
        setTimeout(function() {
          cb(null, 'contents of ' + fileName);
        }, 0);
      }

      var writtenTo = null;
      function writeFile(fileName, text, cb) {
        setTimeout(function () {
          writtenTo = [fileName, text];
          cb();
        }, 0);
      }

      function SuchSubclass(resolve) {
        this._super$constructor(resolve);
      }

      SuchSubclass.prototype = o_create(RSVP.Promise.prototype);
      SuchSubclass.prototype.constructor = SuchSubclass;
      SuchSubclass.prototype._super$constructor = RSVP.Promise;

      var denodeifiedWriteFile = RSVP.denodeify(writeFile);

      var file = new SuchSubclass(function(resolve) {
        readFile('src.txt', function(error, content) {
          if (error) {
            reject(error);
          } else {
            resolve(content);
          }
        });
      });

      denodeifiedWriteFile('dest.txt', file).then(function () {
        assert(objectEquals(writtenTo, ['dest.txt', 'contents of src.txt']));
        done();
      });
    });
  });

  describe("RSVP.hash", function() {
    it('should exist', function() {
      assert(RSVP.hash);
    });

    specify('fulfilled only after all of the promise values are fulfilled', function(done) {
      var firstResolved, secondResolved, firstResolver, secondResolver;

      var first = new RSVP.Promise(function(resolve) {
        firstResolver = resolve;
      });
      first.then(function() {
        firstResolved = true;
      });

      var second = new RSVP.Promise(function(resolve) {
        secondResolver = resolve;
      });
      second.then(function() {
        secondResolved = true;
      });

      setTimeout(function() {
        firstResolver(true);
      }, 0);

      setTimeout(function() {
        secondResolver(true);
      }, 0);

      RSVP.hash({
        first: first,
        second: second
      }).then(function(values) {
        assert(values.first);
        assert(values.second);
        done();
      });
    });

    specify('rejected as soon as a promise is rejected', function(done) {
      var firstResolver, secondResolver;

      var first = new RSVP.Promise(function(resolve, reject) {
        firstResolver = { resolve: resolve, reject: reject };
      });

      var second = new RSVP.Promise(function(resolve, reject) {
        secondResolver = { resolve: resolve, reject: reject };
      });

      setTimeout(function() {
        firstResolver.reject({});
      }, 0);

      var firstWasRejected, secondCompleted;

      first['catch'](function(){
        firstWasRejected = true;
      });

      second['finally'](function(){
        secondCompleted = true;
      });

      RSVP.hash({
        first: first,
        second: second
      }).then(function() {
        assert(false);
        done();
      }, function() {
        assert(firstWasRejected);
        assert(!secondCompleted);
        done();
      });
    });

    specify('resolves an empty hash passed to RSVP.hash()', function(done) {
      RSVP.hash({}).then(function(results) {
        assert(objectEquals(results, {}), 'expected fulfillment');
        done();
      });
    });

    specify('works with null', function(done) {
      RSVP.hash({foo: null}).then(function(results) {
        assert(objectEquals(results.foo, null));
        done();
      });
    });

    specify('works with a truthy value', function(done) {
      RSVP.hash({foo: 1}).then(function(results) {
        assert(objectEquals(results.foo, true));
        done();
      });
    });

    specify('works with a mix of promises and thenables and non-promises', function(done) {
      var promise = new RSVP.Promise(function(resolve) {
        resolve(1);
      });

      var syncThenable = {
        then: function (onFulfilled) {
          onFulfilled(2);
        }
      };

      var asyncThenable = {
        then: function (onFulfilled) {
          setTimeout(function() {
            onFulfilled(3);
          }, 0);
        }
      };

      var nonPromise = 4;

      RSVP.hash({
        promise: promise,
        syncThenable: syncThenable,
        asyncThenable: asyncThenable,
        nonPromise: nonPromise
      }).then(function(results) {
        assert(objectEquals(results, {
          promise: 1,
          syncThenable: 2,
          asyncThenable: 3,
          nonPromise: 4
        }));
        done();
      });
    });

  });

  describe("RSVP.hashSettled", function() {
    it('should exist', function() {
      assert(RSVP.hashSettled);
    });

    specify('fulfilled only after all of the promise values are fulfilled', function(done) {
      var firstResolved, secondResolved, firstResolver, secondResolver;

      var first = new RSVP.Promise(function(resolve) {
        firstResolver = resolve;
      });
      first.then(function() {
        firstResolved = true;
      });

      var second = new RSVP.Promise(function(resolve) {
        secondResolver = resolve;
      });
      second.then(function() {
        secondResolved = true;
      });

      setTimeout(function() {
        firstResolver(true);
      }, 0);

      setTimeout(function() {
        secondResolver(true);
      }, 0);

      RSVP.hashSettled({ first: first, second: second }).then(function(values) {
        assert(values.first);
        assert(values.second);
        done();
      });
    });

    specify('fulfills correctly when constituent promises reject', function(done) {
      var rejectedPromise = new RSVP.Promise(function(resolve, reject) {
        reject(new Error('WHOOPS'));
      });
      var entries = { rejectedPromise: rejectedPromise };
      RSVP.hashSettled(entries).then(function(results) {
        assert(results.rejectedPromise.state, 'rejected' );
        assert(results.rejectedPromise.reason.message, 'WHOOPS' );
        done();
      }, function(reason) {
        assert(false);
        done();
      });
    });

    specify('resolves an empty hash passed to RSVP.hashSettled()', function(done) {
      RSVP.hashSettled({}).then(function(results) {
        assert(objectEquals(results, {}), 'expected fulfillment');
        done();
      });
    });

    specify('works with null', function(done) {
      RSVP.hashSettled({foo: null}).then(function(results) {
        assert(objectEquals(results.foo, {state: 'fulfilled', value: null} ));
        done();
      });
    });

    specify('works with a truthy value', function(done) {
      RSVP.hashSettled({foo: 1}).then(function(results) {
        assert(objectEquals(results.foo, {state: 'fulfilled', value: true} ));
        done();
      });
    });

    specify('works with promises, thenables, non-promises and rejected promises', function(done) {
      var promise = new RSVP.Promise(function(resolve) { resolve(1); });
      var syncThenable = { then: function (onFulfilled) { onFulfilled(2); } };
      var asyncThenable = { then: function (onFulfilled) { setTimeout(function() { onFulfilled(3); }, 0); } };
      var nonPromise = 4;
      var rejectedPromise = new RSVP.Promise(function(resolve, reject) {
        reject(new Error('WHOOPS'));
      });

      var entries = {
        promise: promise,
        syncThenable: syncThenable,
        asyncThenable: asyncThenable,
        nonPromise: nonPromise,
        rejectedPromise: rejectedPromise
      };

      RSVP.hashSettled(entries).then(function(results) {
        assert(objectEquals(results.promise,       {state: 'fulfilled', value: 1} ));
        assert(objectEquals(results.syncThenable,  {state: 'fulfilled', value: 2} ));
        assert(objectEquals(results.asyncThenable, {state: 'fulfilled', value: 3} ));
        assert(objectEquals(results.nonPromise,    {state: 'fulfilled', value: 4} ));
        assert(results.rejectedPromise.state, 'rejected' );
        assert(results.rejectedPromise.reason.message, 'WHOOPS' );
        done();
      });
    });
  });


  describe("RSVP.all", function() {
    testAll(RSVP.all);
  });

  describe("RSVP.Promise.all", function() {
    testAll(function(){
      return RSVP.Promise.all.apply(RSVP.Promise, arguments);
    });
  });

  function testAll(all) {
    it('should exist', function() {
      assert(all);
    });

    it('throws when not passed an array', function(done) {
      var nothing = assertRejection(all());
      var string  = assertRejection(all(''));
      var object  = assertRejection(all({}));

      RSVP.Promise.all([
        nothing,
        string,
        object
      ]).then(function(){ done(); });
    });

    specify('fulfilled only after all of the other promises are fulfilled', function(done) {
      var firstResolved, secondResolved, firstResolver, secondResolver;

      var first = new RSVP.Promise(function(resolve) {
        firstResolver = resolve;
      });
      first.then(function() {
        firstResolved = true;
      });

      var second = new RSVP.Promise(function(resolve) {
        secondResolver = resolve;
      });
      second.then(function() {
        secondResolved = true;
      });

      setTimeout(function() {
        firstResolver(true);
      }, 0);

      setTimeout(function() {
        secondResolver(true);
      }, 0);

      all([first, second]).then(function() {
        assert(firstResolved);
        assert(secondResolved);
        done();
      });
    });

    specify('rejected as soon as a promise is rejected', function(done) {
      var firstResolver, secondResolver;

      var first = new RSVP.Promise(function(resolve, reject) {
        firstResolver = { resolve: resolve, reject: reject };
      });

      var second = new RSVP.Promise(function(resolve, reject) {
        secondResolver = { resolve: resolve, reject: reject };
      });

      setTimeout(function() {
        firstResolver.reject({});
      }, 0);

      var firstWasRejected, secondCompleted;

      first['catch'](function(){
        firstWasRejected = true;
      });

      second['finally'](function(){
        secondCompleted = true;
      });

      all([first, second]).then(function() {
        assert(false);
      }, function() {
        assert(firstWasRejected);
        assert(!secondCompleted);
        done();
      });
    });

    specify('passes the resolved values of each promise to the callback in the correct order', function(done) {
      var firstResolver, secondResolver, thirdResolver;

      var first = new RSVP.Promise(function(resolve, reject) {
        firstResolver = { resolve: resolve, reject: reject };
      });

      var second = new RSVP.Promise(function(resolve, reject) {
        secondResolver = { resolve: resolve, reject: reject };
      });

      var third = new RSVP.Promise(function(resolve, reject) {
        thirdResolver = { resolve: resolve, reject: reject };
      });

      thirdResolver.resolve(3);
      firstResolver.resolve(1);
      secondResolver.resolve(2);

      all([first, second, third]).then(function(results) {
        assert(results.length === 3);
        assert(results[0] === 1);
        assert(results[1] === 2);
        assert(results[2] === 3);
        done();
      });
    });

    specify('resolves an empty array passed to all()', function(done) {
      all([]).then(function(results) {
        assert(results.length === 0);
        done();
      });
    });

    specify('works with null', function(done) {
      all([null]).then(function(results) {
        assert.equal(results[0], null);
        done();
      });
    });

    specify('works with a mix of promises and thenables and non-promises', function(done) {
      var promise = new RSVP.Promise(function(resolve) { resolve(1); });
      var syncThenable = { then: function (onFulfilled) { onFulfilled(2); } };
      var asyncThenable = { then: function (onFulfilled) { setTimeout(function() { onFulfilled(3); }, 0); } };
      var nonPromise = 4;

      all([promise, syncThenable, asyncThenable, nonPromise]).then(function(results) {
        assert(objectEquals(results, [1, 2, 3, 4]));
        done();
      })['catch'](done);
    });
  }

  describe("RSVP.allSettled", function() {
    it('should exist', function() {
      assert(RSVP.allSettled);
    });

    it('throws when not passed an array', function(done) {
      var nothing = assertRejection(RSVP.allSettled());
      var string  = assertRejection(RSVP.allSettled(''));
      var object  = assertRejection(RSVP.allSettled({}));

      RSVP.Promise.all([
        nothing,
        string,
        object
      ]).then(function(){ done(); });
    });

    specify('resolves an empty array passed to allSettled()', function(done) {
      RSVP.allSettled([]).then(function(results) {
        assert(results.length === 0);
        done();
      });
    });

    specify('works with promises, thenables, non-promises and rejected promises',
      function(done) {
        var promise = new RSVP.Promise(function(resolve) { resolve(1); });
        var syncThenable = { then: function (onFulfilled) { onFulfilled(2); } };
        var asyncThenable = {
          then: function (onFulfilled) {
            setTimeout(function() { onFulfilled(3); }, 0);
          }
        };
        var nonPromise = 4;
        var rejectedPromise = new RSVP.Promise(function(resolve, reject) {
          reject(new Error('WHOOPS'));
        });

        var entries = new Array(
          promise, syncThenable, asyncThenable, nonPromise, rejectedPromise
        );

        RSVP.allSettled(entries).then(function(results) {
          assert(objectEquals(results[0], {state: "fulfilled", value: 1} ));
          assert(objectEquals(results[1], {state: "fulfilled", value: 2} ));
          assert(objectEquals(results[2], {state: "fulfilled", value: 3} ));
          assert(objectEquals(results[3], {state: "fulfilled", value: 4} ));
          assert(results[4].state, "rejected");
          assert(results[4].reason.message, "WHOOPS");
          done();
        });
      }
    );

    specify('fulfilled only after all of the other promises are fulfilled', function(done) {
      var firstResolved, secondResolved, firstResolver, secondResolver;

      var first = new RSVP.Promise(function(resolve) {
        firstResolver = resolve;
      });
      first.then(function() {
        firstResolved = true;
      });

      var second = new RSVP.Promise(function(resolve) {
        secondResolver = resolve;
      });
      second.then(function() {
        secondResolved = true;
      });

      setTimeout(function() {
        firstResolver(true);
      }, 0);

      setTimeout(function() {
        secondResolver(true);
      }, 0);

      RSVP.allSettled([first, second]).then(function() {
        assert(firstResolved);
        assert(secondResolved);
        done();
      });
    });

  });

  describe("RSVP.reject", function(){
    specify("it should exist", function(){
      assert(RSVP.reject);
    });

    describe('it rejects', function(){
      var reason = 'the reason',
      promise = RSVP.reject(reason);

      promise.then(function(){
        assert(false, 'should not fulfill');
      }, function(actualReason){
        assert.equal(reason, actualReason);
      });
    });
  });

  describe("RSVP.on", function(){
    afterEach(function() {
      RSVP.off('error');
    });

    it("can be off'd", function(){
      var broCount = 0;
      var obj = {};
      RSVP.EventTarget.mixin(obj);

      function brohandler(){
        broCount++;
      }

      obj.on('bro', brohandler);

      assert.equal(broCount, 0, 'bro was only trigged once');

      obj.trigger('bro');

      assert.equal(broCount, 1, 'bro was only trigged once');

      obj.off('bro', brohandler);
      obj.trigger('bro');

      assert.equal(broCount, 1, 'bro was only trigged once');
    });

    it("When provided, any unhandled exceptions are sent to it", function(done) {
      var thrownError = new Error();

      RSVP.on('error', function(reason) {
        assert.equal(reason, thrownError, "The thrown error is passed in");
        done();
      });

      new RSVP.Promise(function(resolve, reject) {
        reject(thrownError);
      });
    });

    it("When provided, handled exceptions are not sent to it", function(done) {
      var thrownError = new Error();

      RSVP.on('error', function(event) {
        assert(false, "Should not get here");
      });

      var promise = new RSVP.Promise(function(resolve, reject) {
        reject(thrownError);
      });

      promise.then(null, function(error) {
        assert.equal(error, thrownError, "The handler should handle the error");
        done();
      });
    });

    it("finally should not 'handle' errors (1)", function(done) {
      var thrownError = new Error();

      RSVP.on('error', function(event) {
        assert(true, 'Should get here');
        done();
      });

      var promise = new RSVP.Promise(function(resolve, reject) {
        reject(thrownError);
      });

      promise.finally(function(error) {

      });
    });

    it("finally should not 'handle' errors (2)", function(done) {
      var thrownError = new Error();

      RSVP.on('error', function(event) {
        assert(true, 'Should get here');
        done();
      });

      var promise = new RSVP.Promise(function(resolve, reject) {
        reject(thrownError);
      });

      promise.then().finally(function(error) {

      });
    });

    it("finally should not 'handle' errors (3)", function(done) {
      var thrownError = new Error();

      RSVP.on('error', function(event) {
        assert(false, 'Should not get here');
        done();
      });

      var promise = new RSVP.Promise(function(resolve, reject) {
        reject(thrownError);
      });

      promise.then().finally(function(error) {

      }).then(null, function() {
        assert(true);
        done();
      });
    });

    it("finally should trigger onError when its callback throws", function(done) {
      var thrownError = new Error();
      var finallyError = new Error("OMG");

      RSVP.on('error', function(event) {
        assert(true, 'Should get here');
        assert.equal(finallyError, event, "correct error is triggered");
        done();
      });

      var promise = new RSVP.Promise(function(resolve, reject) {
        reject(thrownError);
      });

      promise.then().finally(function(error) {
        throw finallyError;
      });
    });

    it("finally should trigger onError when its callback rejects", function(done) {
      var thrownError = new Error();
      var finallyError = new Error("OMG");

      RSVP.on('error', function(event) {
        assert(true, 'Should get here');
        assert.equal(finallyError, event, "correct error is triggered");
        done();
      });

      var promise = new RSVP.Promise(function(resolve, reject) {
        reject(thrownError);
      });

      promise.then().finally(function(error) {
        return RSVP.reject(finallyError);
      });
    });

  });

  function assertRejection(promise) {
    return promise.then(function(){
      assert(false, 'expected rejection, but got fulfillment');
    }, function(reason){
      assert(reason instanceof Error);
    });
  }

  function testRace(race) {
    it("should exist", function() {
      assert(race);
    });

    it("throws when not passed an array", function(done) {
      var nothing = assertRejection(race());
      var string  = assertRejection(race(''));
      var object  = assertRejection(race({}));

      RSVP.Promise.all([
        nothing,
        string,
        object
      ]).then(function(){ done(); });
    });

    specify('fulfilled after one of the other promises are fulfilled', function(done) {
      var firstResolved, secondResolved, firstResolver, secondResolver;

      var first = new RSVP.Promise(function(resolve) {
        firstResolver = resolve;
      });
      first.then(function() {
        firstResolved = true;
      });

      var second = new RSVP.Promise(function(resolve) {
        secondResolver = resolve;
      });
      second.then(function() {
        secondResolved = true;
      });

      setTimeout(function() {
        firstResolver(true);
      }, 100);

      setTimeout(function() {
        secondResolver(true);
      }, 0);

      race([first, second]).then(function() {
        assert(secondResolved);
        assert.equal(firstResolved, undefined);
        done();
      });
    });

    specify('the race begins on nextTurn and prioritized by array entry', function(done) {
      var firstResolver, secondResolver, nonPromise = 5;

      var first = new RSVP.Promise(function(resolve, reject) {
        resolve(true);
      });

      var second = new RSVP.Promise(function(resolve, reject) {
        resolve(false);
      });

      race([first, second, nonPromise]).then(function(value) {
        assert.equal(value, true);
        done();
      });
    });

    specify('rejected as soon as a promise is rejected', function(done) {
      var firstResolver, secondResolver;

      var first = new RSVP.Promise(function(resolve, reject) {
        firstResolver = { resolve: resolve, reject: reject };
      });

      var second = new RSVP.Promise(function(resolve, reject) {
        secondResolver = { resolve: resolve, reject: reject };
      });

      setTimeout(function() {
        firstResolver.reject({});
      }, 0);

      var firstWasRejected, secondCompleted;

      first['catch'](function(){
        firstWasRejected = true;
      });

      second['finally'](function(){
        secondCompleted = true;
      });

      race([first, second]).then(function() {
        assert(false);
      }, function() {
        assert(firstWasRejected);
        assert(!secondCompleted);
        done();
      });
    });

    specify('resolves an empty array to forever pending Promise', function(done) {
      var foreverPendingPromise = RSVP.race([]),
          wasSettled            = false;

      foreverPendingPromise.then(function() {
        wasSettled = true;
      }, function() {
        wasSettled = true;
      });

      setTimeout(function() {
        assert(!wasSettled);
        done();
      }, 100);
    });

    specify('works with a mix of promises and thenables', function(done) {
      var promise = new RSVP.Promise(function(resolve) { setTimeout(function() { resolve(1); }, 10); }),
          syncThenable = { then: function (onFulfilled) { onFulfilled(2); } };

      race([promise, syncThenable]).then(function(result) {
        assert(result, 2);
        done();
      });
    });

    specify('works with a mix of thenables and non-promises', function (done) {
      var asyncThenable = { then: function (onFulfilled) { setTimeout(function() { onFulfilled(3); }, 0); } },
          nonPromise = 4;
      race([asyncThenable, nonPromise]).then(function(result) {
        assert(result, 4);
        done();
      });
    });
  }

  describe("RSVP.race", function() {
    testRace(RSVP.race);
  });

  describe("RSVP.Promise.race", function() {
    testRace(function(){
      return RSVP.Promise.race.apply(RSVP.Promise, arguments);
    });
  });

  describe("RSVP.onerror", function(){
    var onerror;

    afterEach(function() {
      RSVP.off('error');
    });

    it("When provided, any unhandled exceptions are sent to it (false positive case)", function(done) {
      var thrownError = new Error();

      RSVP.configure('onerror', function(error) {
        assert.equal(error, thrownError, "The thrown error is passed in");
        done();
      });

      new RSVP.Promise(function(resolve, reject) {
        reject(thrownError);
      }).then(function() {
        // doesn't get here
        assert(false);
      });
    });

    it("When provided, handled exceptions are not sent to it", function(done) {
      var thrownError = new Error();

      RSVP.configure('onerror', function(error) {
        assert(false, "Should not get here");
      });

      new RSVP.Promise(function(resolve, reject) {
        reject(thrownError);
      }).then(null, function(error) {
        assert.equal(error, thrownError, "The handler should handle the error");
        done();
      });
    });

    it("Enumerator: When provided, any unhandled exceptions are sent to it", function(done) {
      var thrownError = new Error();

      RSVP.configure('onerror', function(error) {
        assert.equal(error, thrownError, "The thrown error is passed in");
        done();
      });

      var promise = new RSVP.Promise(function(resolve, reject) {
        reject(thrownError);
      });

      RSVP.Promise.all([promise]).then(function() {
        // doesn't get here
        assert(false);
      });
    });

    it("all When provided, handled exceptions are not sent to it", function(done) {
      var thrownError = new Error();

      RSVP.configure('onerror', function(error) {
        assert(false, "Should not get here");
      });

      var promise = new RSVP.Promise(function(resolve, reject) {
        reject(thrownError);
      });

      RSVP.Promise.all([promise]).then(undefined, function(reason) {
        assert.equal(reason, thrownError, "The handler should handle the error");
        done();
      });
    });

    it("assimilation: When provided, handled exceptions are not sent to it", function(done) {
      var thrownError = new Error();

      RSVP.configure('onerror', function(error) {
        assert(false, "Should not get here");
      });

      RSVP.Promise.resolve().then(function(){
        return RSVP.Promise.reject(thrownError);
      }).then(undefined, function(reason) {
        assert.equal(reason, thrownError, "The handler should handle the error");
        done();
      });
    });

    it("zalgo: When provided, unhandled exceptions are sent to it", function(done) {
      var thrownError = new Error();

      RSVP.configure('onerror', function(error) {
        assert(true, "should also have emitted onerror");
        assert.equal(error, thrownError, "The handler should handle the error");
        done();
      });

      var rejected = RSVP.Promise.reject(thrownError);

      RSVP.Promise.resolve().then(function(){
        return rejected;
      });
    });


    it("assimilation: When provided, unhandled exceptions are sent to it", function(done) {
      var thrownError = new Error();

      RSVP.configure('onerror', function(error) {
        assert(true, "Should get here once");
        done();
      });

      var promise = new RSVP.Promise(function(resolve, reject) {
        reject(thrownError);
      });

      RSVP.Promise.resolve().then(function(){
        return promise;
      });
    });

    it("When provided, any unhandled exceptions are sent to it, even from inside error handlers", function(done) {
      var firstError = new Error();
      var secondError = new Error();

      RSVP.on('error', function(reason) {
        assert.equal(reason, secondError, "The thrown error is passed in");
        done();
      });

      new RSVP.Promise(function(resolve, reject) {
        reject(firstError);
      }).catch(function(err) {
        assert.equal(err, firstError, "The first error is handled");
        return new RSVP.Promise(function(resolve, reject) {
          reject(secondError);
        });
      });
    });

    it('does not emit if handled by the end of the current turn', function(done){
      var rejectionCount = 0;
      var reason = new Error('Rejection Reason');

      RSVP.on('error', function(event){
        assert(false, 'should not have errored');
        done();
      });

      var promise = RSVP.Promise.reject(reason);
      RSVP.Promise.resolve().then(function() {
        promise.catch(function() {
          done();
        });
      });
    });

  });

  describe("RSVP.resolve", function(){
    specify("it should exist", function(){
      assert(RSVP.resolve);
    });

    describe("1. If x is a promise, adopt its state ", function(){
      specify("1.1 If x is pending, promise must remain pending until x is fulfilled or rejected.", function(done){
        var expectedValue, resolver, thenable, wrapped;

        expectedValue = 'the value';
        thenable = {
          then: function(resolve, reject){
            resolver = resolve;
          }
        };

        wrapped = RSVP.resolve(thenable);

        wrapped.then(function(value){
          assert(value === expectedValue);
          done();
        });

        setTimeout(function(){
          resolver(expectedValue);
        }, 10);
      });

      specify("1.2 If/when x is fulfilled, fulfill promise with the same value.", function(done){
        var expectedValue, thenable, wrapped;

        expectedValue = 'the value';
        thenable = {
          then: function(resolve, reject){
            resolve(expectedValue);
          }
        };

        wrapped = RSVP.resolve(thenable);

        wrapped.then(function(value){
          assert(value === expectedValue);
          done();
        })
      });

      specify("1.3 If/when x is rejected, reject promise with the same reason.", function(done){
        var expectedError, thenable, wrapped;

        expectedError =  new Error();
        thenable = {
          then: function(resolve, reject){
            reject(expectedError);
          }
        };

        wrapped = RSVP.resolve(thenable);

        wrapped.then(null, function(error){
          assert(error === expectedError);
          done();
        });
      });
    });

    describe("2. Otherwise, if x is an object or function,", function(){
      specify("2.1 Let then x.then", function(done){
        var accessCount, resolver, wrapped, thenable;

        accessCount = 0;
        thenable = { };

        // we likely don't need to test this, if the browser doesn't support it
        if (typeof Object.defineProperty !== "function") { done(); return; }

        Object.defineProperty(thenable, 'then', {
          get: function(){
            accessCount++;

            if (accessCount > 1) {
              throw new Error();
            }

            return function(){ };
          }
        });

        assert(accessCount === 0);

        wrapped = RSVP.resolve(thenable);

        assert(accessCount === 1);

        done();
      });

      specify("2.2 If retrieving the property x.then results in a thrown exception e, reject promise with e as the reason.", function(done){
        var wrapped, thenable, expectedError;

        expectedError = new Error();
        thenable = { };

        // we likely don't need to test this, if the browser doesn't support it
        if (typeof Object.defineProperty !== "function") { done(); return; }

        Object.defineProperty(thenable, 'then', {
          get: function(){
            throw expectedError;
          }
        });

        wrapped = RSVP.resolve(thenable);

        wrapped.then(null, function(error){
          assert(error === expectedError, 'incorrect exception was thrown');
          done();
        });
      });

      describe('2.3. If then is a function, call it with x as this, first argument resolvePromise, and second argument rejectPromise, where', function(){
        specify('2.3.1 If/when resolvePromise is called with a value y, run Resolve(promise, y)', function(done){
          var expectedSuccess, resolver, rejector, thenable, wrapped, calledThis;

          thenable = {
            then: function(resolve, reject){
              calledThis = this;
              resolver = resolve;
              rejector = reject;
            }
          };

          expectedSuccess = 'success';
          wrapped = RSVP.resolve(thenable);

          wrapped.then(function(success){
            assert(calledThis === thenable, 'this must be the thenable');
            assert(success === expectedSuccess, 'rejected promise with x');
            done();
          });

          setTimeout(function() {
            resolver(expectedSuccess);
          }, 20);
        });

        specify('2.3.2 If/when rejectPromise is called with a reason r, reject promise with r.', function(done){
          var expectedError, resolver, rejector, thenable, wrapped, calledThis;

          thenable = {
            then: function(resolve, reject){
              calledThis = this;
              resolver = resolve;
              rejector = reject;
            }
          };

          expectedError = new Error();

          wrapped = RSVP.resolve(thenable);

          wrapped.then(null, function(error){
            assert(error === expectedError, 'rejected promise with x');
            done();
          });

          setTimeout(function() {
            rejector(expectedError);
          }, 20);
        });

        specify("2.3.3 If both resolvePromise and rejectPromise are called, or multiple calls to the same argument are made, the first call takes precedence, and any further calls are ignored", function(done){
          var expectedError, expectedSuccess, resolver, rejector, thenable, wrapped, calledThis,
          calledRejected, calledResolved;

          calledRejected = 0;
          calledResolved = 0;

          thenable = {
            then: function(resolve, reject){
              calledThis = this;
              resolver = resolve;
              rejector = reject;
            }
          };

          expectedError = new Error();

          wrapped = RSVP.resolve(thenable);

          wrapped.then(function(){
            calledResolved++;
          }, function(error){
            calledRejected++;
            assert(calledResolved === 0, 'never resolved');
            assert(calledRejected === 1, 'rejected only once');
            assert(error === expectedError, 'rejected promise with x');
          });

          setTimeout(function() {
            rejector(expectedError);
            rejector(expectedError);

            rejector('foo');

            resolver('bar');
            resolver('baz');
          }, 20);

          setTimeout(function(){
            assert(calledRejected === 1, 'only rejected once');
            assert(calledResolved === 0, 'never resolved');
            done();
          }, 50);
        });

        describe("2.3.4 If calling then throws an exception e", function(){
          specify("2.3.4.1 If resolvePromise or rejectPromise have been called, ignore it.", function(done){
            var expectedSuccess, resolver, rejector, thenable, wrapped, calledThis,
            calledRejected, calledResolved;

            expectedSuccess = 'success';

            thenable = {
              then: function(resolve, reject){
                resolve(expectedSuccess);
                throw expectedError;
              }
            };

            wrapped = RSVP.resolve(thenable);

            wrapped.then(function(success){
              assert(success === expectedSuccess, 'resolved not errored');
              done();
            });
          });

          specify("2.3.4.2 Otherwise, reject promise with e as the reason.", function(done) {
            var expectedError, resolver, rejector, thenable, wrapped, calledThis, callCount;

            expectedError = new Error();
            callCount = 0;

            thenable = { then: function() { throw expectedError; } };

            wrapped = RSVP.resolve(thenable);

            wrapped.then(null, function(error){
              callCount++;
              assert(expectedError === error, 'expected the correct error to be rejected');
              done();
            });

            assert(callCount === 0, 'expected async, was sync');
          });
        });
      });

      specify("2.4 If then is not a function, fulfill promise with x", function(done){
        var expectedError, resolver, rejector, thenable, wrapped, calledThis, callCount;

        thenable = { then: 3 };
        callCount = 0;
        wrapped = RSVP.resolve(thenable);

        wrapped.then(function(success){
          callCount++;
          assert(thenable === success, 'fulfilled promise with x');
          done();
        });

        assert(callCount === 0, 'expected async, was sync');
      });
    });

    describe("3. If x is not an object or function, ", function(){
      specify("fulfill promise with x.", function(done){
        var thenable, callCount, wrapped;

        thenable = null;
        callCount = 0;
        wrapped = RSVP.resolve(thenable);

        wrapped.then(function(success){
          callCount++;
          assert(success === thenable, 'fulfilled promise with x');
          done();
        }, function(a){
          assert(false, 'should not also reject');
        });

        assert(callCount === 0, 'expected async, was sync');
      });
    });
  });

  describe("inspection", function(){
    beforeEach(function () {
      RSVP.configure('instrument', true);
    });

    afterEach(function(){
      RSVP.configure('instrument', false);
    });

    describe("when `instrument` is falsy", function () {
      beforeEach(function () {
        RSVP.configure('instrument', false);
      });

      afterEach(function(){
        RSVP.off('created');
        RSVP.off('chained');
        RSVP.off('fulfilled');
        RSVP.off('rejected');
      });

      specify("created is not fired", function (done) {
        RSVP.on('created', function () {
          assert(false, 'created erroneously fired');
        });

        RSVP.resolve().then(done);
      });

      specify("chained is not fired", function (done) {
        RSVP.on('chained', function () {
          assert(false, 'chained erroneously fired');
        });

        RSVP.resolve().then(done);
      });

      specify("fulfilled is not fired", function (done) {
        RSVP.on('fulfilled', function () {
          assert(false, 'fulfilled erroneously fired');
        });

        RSVP.resolve().then(done);
      });

      specify("rejected is not fired", function (done) {
        RSVP.on('rejected', function () {
          assert(false, 'rejected erroneously fired');
        });

        RSVP.reject()['catch'](done);
      });
    });

    function parseGuid(guid){
      var matches = guid.match(/rsvp_(\d+)-(\d+)/);

      if (matches) {
        return {
          key: matches[1],
          index: parseInt(matches[2], 10)
        };
      } else {
        throw new Error('unknown guid:' + guid);
      }
    }

    describe("creation", function(){
      afterEach(function(){
        RSVP.off('created');
        RSVP.configure('instrument', false);
      });

      beforeEach(function () {
        RSVP.configure('instrument', true);
      });

      specify("it emits a creation event", function(done){
        var creationCount = 0;
        var promise;
        var deferred;

        RSVP.on('created', function(event){
          assert(true, 'event was called');
          assert.equal(creationCount, 0 ,'creation event was only emitted once');
          creationCount++;

          var parsedGuid = parseGuid(event.guid);

          assert(event.guid, 'has a guid');
          assert(parsedGuid.key, 'has a key');
          assert(parsedGuid.index, 'has a count');

          assert.equal(event.eventName, 'created');
          assert(event.timeStamp, 'has a timeStamp');
          assert.ok(!event.detail, 'a created event has no detail');

          done();
        });

        deferred = RSVP.defer();
        promise = deferred.promise;
      });

      specify("it emits a unique guid", function(done){
        var creationCount = 0;
        var lastIndex = 0;

        RSVP.on('created', function(event){
          assert(true, 'event was called');
          creationCount++;
          var parsedGuid = parseGuid(event.guid);

          if (creationCount > 1) {
            assert.equal(parsedGuid.index, lastIndex + 1, 'incrementing guid');
            done();
          }

          lastIndex= parsedGuid.index;
        });

        RSVP.defer();
        RSVP.defer();
      });
    });

    describe("rejection", function(done){
      afterEach(function(){
        RSVP.off('rejected');
      });

      specify('emits rejection event', function(done){
        var rejectionCount = 0;
        var reason = new Error('Rejection Reason');

        RSVP.on('rejected', function(event){
          rejectionCount++;
          assert.equal(rejectionCount, 1, 'emitted the rejection event only once');

          assert.equal(event.eventName, 'rejected');
          assert(event.guid, 'has a guid');
          assert.equal(event.detail, reason, 'correct rejection reason');
          assert(event.timeStamp, 'has a timeStamp');

          done();
        });

        var promise = RSVP.reject(reason);
      });
    });

    describe("fulfillment", function(){
      afterEach(function(){
        RSVP.off('fulfilled');
      });

      specify('emits fulfillment event', function(done){
        var fulfillmentCount= 0;
        var value = 'fulfillment value';

        RSVP.on('fulfilled', function(event){
          fulfillmentCount++;
          assert.equal(fulfillmentCount, 1, 'emitted the fulfilment event only once');

          assert.equal(event.eventName, 'fulfilled');
          assert(event.guid, 'has a guid');
          assert.equal(event.detail, value, 'correct fulfillment value');
          assert(event.timeStamp, 'has a timeStamp');

          done();
        });

        var promise = RSVP.resolve(value);
      });
    });

    describe("chained", function(){
      afterEach(function(){
        RSVP.off('chained');
      });

      specify('emits chained event', function(done) {
        var value = 'fulfillment value';
        var promise = RSVP.resolve(value);

        RSVP.on('chained', function(event){
          var parent = event.guid;
          var child = event.childGuid;

          assert(parent, 'has parent');
          assert(child, 'has child');

          var parsedParent = parseGuid(parent);
          var parsedChild  = parseGuid(child);

          assert(parsedParent.key);
          assert(parsedParent.index);

          assert.equal(event.guid, promise._guidKey + promise._id, 'has correct parent reference');

          assert(parsedChild.key);
          assert(parsedChild.index);

          assert.equal(event.eventName, 'chained');
          assert(event.timeStamp, 'has a timeStamp');

          done();
        });

        promise.then();
      });
    });
  });

  describe("RSVP.async", function() {

    var values, originalAsync;
    beforeEach(function() {
      originalAsync = RSVP.configure('async');
      values = [];
    });

    afterEach(function() {
      RSVP.configure('async', originalAsync);
    });

    function append(value) {
      values.push(value);
    }

    function runTest(done) {
      RSVP.resolve(1).then(append);

      RSVP.async(function(value) {
        assert.deepEqual(values, [1]);
        values.push(value);
      }, 2);

      RSVP.resolve(3).then(append);

      RSVP.async(function() {
        assert.deepEqual(values, [1,2,3]);
        if (done) {
          done();
        }
      });
    }

    //it("schedules items to RSVP's internal queue", function(done) {
    //  runTest(done);
    //});

    it("can be configured to use a different schedule via configure('async', fn)", function() {
      var actions = [];
      RSVP.configure('async', function(callback, promise) {
        actions.push([callback, promise]);
      });

      runTest();

      // Flush custom deferred action queue
      while(actions.length) {
        var action = actions.shift();
        action[0](action[1]);
      }
    });
  });

  describe("Promise.finally", function() {
    describe("native finally behaviour", function() {
      describe("no value is passed in", function() {
        it("does not provide a value to the finally code", function(done) {
          var fulfillmentValue = 1;
          var promise = RSVP.resolve(fulfillmentValue);

          promise['finally'](function() {
            assert.equal(arguments.length, 0);
            done();
          });
        });

        it("does not provide a reason to the finally code", function(done) {
          var rejectionReason = new Error();
          var promise = RSVP.reject(rejectionReason);

          promise['finally'](function(arg) {
            assert.equal(arguments.length, 0);
            done();
          });
        });
      });

      describe("non-exceptional cases do not affect the result", function(){
        it("preserves the original fulfillment value even if the finally callback returns a value", function(done) {
          var fulfillmentValue = 1;
          var promise = RSVP.resolve(fulfillmentValue);

          promise['finally'](function() {
            return 2;
          }).then(function(value) {
            assert.equal(fulfillmentValue, value);
            done();
          });
        });

        it("preserves the original rejection reason even if the finally callback returns a value", function(done) {
          var rejectionReason = new Error();
          var promise = RSVP.reject(rejectionReason);

          promise['finally'](function() {
            return 2;
          }).then(undefined, function(reason) {
            assert.equal(rejectionReason, reason);
            done();
          });
        });
      });

      describe("exception cases do propogate the failure", function(){
        describe("fulfilled promise", function(){
          it("propagates changes via throw", function(done) {
            var promise = RSVP.resolve(1);
            var expectedReason  = new Error();

            promise['finally'](function() {
              throw expectedReason;
            }).then(undefined, function(reason) {
              assert.deepEqual(expectedReason, reason);
              done();
            });
          });

          it("propagates changes via returned rejected promise", function(done){
            var promise = RSVP.resolve(1);
            var expectedReason  = new Error();

            promise['finally'](function() {
              return RSVP.reject(expectedReason);
            }).then(undefined, function(reason) {
              assert.deepEqual(expectedReason, reason);
              done();
            });
          });
        });

        describe("rejected promise", function(){
          it("propagates changes via throw", function(done) {
            var promise = RSVP.reject(1);
            var expectedReason  = new Error();

            promise['finally'](function() {
              throw expectedReason;
            }).then(undefined, function(reason) {
              assert.deepEqual(expectedReason, reason);
              done();
            });
          });

          it("propagates changes via returned rejected promise", function(done){
            var promise = RSVP.reject(1);
            var expectedReason  = new Error();

            promise['finally'](function() {
              return RSVP.reject(expectedReason);
            }).then(undefined, function(reason) {
              assert.deepEqual(expectedReason, reason);
              done();
            });
          });
        });
      });
    });

    describe("inheritance", function() {
      function Subclass (resolver) {
        this._promise$constructor(resolver);
      }

      Subclass.prototype = o_create(RSVP.Promise.prototype);
      Subclass.prototype.constructor = Subclass;
      Subclass.prototype._promise$constructor = RSVP.Promise;

      Subclass.resolve = RSVP.Promise.resolve;
      Subclass.reject = RSVP.Promise.reject;
      Subclass.all = RSVP.Promise.all;

      it("preserves correct subclass when chained", function() {
        var promise = Subclass.resolve().finally();
        assert.ok(promise instanceof Subclass);
        assert.equal(promise.constructor, Subclass);
      });

      it("preserves correct subclass when rejected", function() {
        var promise = Subclass.resolve().finally(function() {
          throw new Error("OMG");
        });
        assert.ok(promise instanceof Subclass);
        assert.equal(promise.constructor, Subclass);
      });

      it("preserves correct subclass when someone returns a thenable", function() {
        var promise = Subclass.resolve().finally(function() {
          return RSVP.Promise.resolve(1);
        });
        assert.ok(promise instanceof Subclass);
        assert.equal(promise.constructor, Subclass);
      });
    });
  });

  describe("RSVP.filter", function(){

    var filterFn = function(item){
      return item > 1;
    };

    it("exists", function(){
      assert(RSVP.filter);
    });

    it("returns results that filterFn returns truthy values", function(done){
      var promises = [
        RSVP.resolve(1),
        RSVP.resolve(2),
        RSVP.resolve(3)
      ];

      RSVP.filter(promises, filterFn).then(function(results){
        assert.deepEqual([2, 3], results);
        done();
      }, done);
    });

    it("throws an error if an array is not passed", function(){
      assertRejection(RSVP.filter());
    });

    it("throws an error if a filterFn is not passed", function(){
      assertRejection(RSVP.filter([]));
    });

    it("works with non-promise values and promises", function(done){
      var promises = [
        RSVP.resolve(1),
        RSVP.resolve(2),
        3
      ];

      RSVP.filter(promises, filterFn).then(function(results){
        assert.deepEqual([2, 3], results);
        done();
      },function(reason) {
        done(reason);
      });
    });

    it("waits if filterFn returns a promise", function(done){

      var filterFn = function(item){
        if (item === 1 ) {
          return RSVP.resolve(true);
        }
        return true;
      };
      var promises = [
        RSVP.resolve(1),
        RSVP.resolve(2),
        3
      ];

      var results1 = RSVP.filter(promises, filterFn);

      filterFn = function(item){
        if (item === 1) {
          return RSVP.resolve(false);
        }
        return true;
      }

      var results2 = RSVP.filter(promises, filterFn);

      RSVP.all([results1, results2]).then(function(results){
        var results1 = results[0];
        var results2 = results[1];
        assert.deepEqual(results1, [1, 2, 3]);
        assert.deepEqual(results2, [ 2, 3 ]);
        done();
      }, done);

    });
  });

  describe("RSVP.map", function(){
    var mapFn = function(item){
      return item + 1;
    };

    it("exists", function(){
      assert(RSVP.map);
    });

    it("calls mapFn on all the promises that are resolved", function(done){
      var promise1 = RSVP.resolve(1);
      var promise2 = RSVP.resolve(2);
      var promise3 = RSVP.resolve(3);
      var promises = [ promise1, promise2, promise3 ];

      RSVP.map(promises, mapFn).then(function(results){
        assert.deepEqual([ 2, 3, 4], results);
        done();
      });
    });

    it("throws an error if an array is not passed", function(){
      assertRejection(RSVP.map());
    });

    it("throws an error if a mapFn is not passed", function(){
      assertRejection(RSVP.map([]));
    });

    it("works with non-promise values and promises", function(done){

      var values = [ 1, RSVP.resolve(2) ];

      RSVP.map(values, mapFn).then(function(results){
        assert.deepEqual([2, 3], results);
        done();
      });
    });

    it("becomes rejected with the first promise that becomes rejected", function(done){

      var promises = [
        RSVP.reject(new Error("prefix:1")),
        RSVP.reject(new Error("prefix:2")),
        1
      ];

      RSVP.map(promises, mapFn).then(function(){
        done(new Error("Promise was resolved when it shouldn't have been!"));
      }, function(reason){
        assert(reason.message === "prefix:1");
        done();
      });
    });

    it("waits if a promise is returned from mapFn", function(done){
      var values = [ 1, 2, 3 ];
      var mapFn = function(value){
        return RSVP.resolve(value + 1);
      };

      RSVP.map(values, mapFn).then(function(values){
        assert.deepEqual(values, [ 2, 3, 4 ]);
        done();
      }, done);
    });

    it("becomes rejected if a promise returned from mapFn becomes rejected", function(done){

      var expectedErrorMessage = "must-be-prefixed-with-non-number-for-old-ie:1"
      var values = [ 1, 2, 3 ];
      var mapFn = function(value){
        // http://msdn.microsoft.com/en-us/library/ie/dww53sbt(v=vs.94).aspx
        return RSVP.reject(new Error(expectedErrorMessage));
      };

      RSVP.map(values, mapFn).then(function(){
        done(new Error("Promise should not be resolved!"));
      }, function (reason) {
        assert(reason.message === expectedErrorMessage);
        done();
      });
    });
  });

  if (typeof Worker !== 'undefined') {
    describe('web worker', function () {
      it('should work', function (done) {
        var worker = new Worker('worker.js');
        worker.addEventListener('error', function(reason) {
          done(new Error("Test failed:" + reason));
        });
        worker.addEventListener('message', function (e) {
          worker.terminate();
          assert.equal(e.data, 'pong');
          done();
        });
        worker.postMessage('ping');
      });
    });
  }
});

// thanks to @wizardwerdna for the test case -> https://github.com/tildeio/rsvp.js/issues/66
// Only run these tests in node (phantomjs cannot handle them)
if (typeof module !== 'undefined' && module.exports) {

  describe("using reduce to sum integers using promises", function(){
    var resolve = RSVP.resolve;

    it("should build the promise pipeline without error", function(){
      var array, iters, pZero, i;

      array = [];
      iters = 1000;

      for (i=1; i<=iters; i++) {
        array.push(i);
      }

      pZero = resolve(0);

      array.reduce(function(promise, nextVal) {
        return promise.then(function(currentVal) {
          return resolve(currentVal + nextVal);
        });
      }, pZero);
    });

    it("should get correct answer without blowing the nextTick stack", function(done){
      var pZero, array, iters, result, i;

      pZero = resolve(0);

      array = [];
      iters = 1000;

      for (i=1; i<=iters; i++) {
        array.push(i);
      }

      result = array.reduce(function(promise, nextVal) {
        return promise.then(function(currentVal) {
          return resolve(currentVal + nextVal);
        });
      }, pZero);

      result.then(function(value){
        assert.equal(value, (iters*(iters+1)/2));
        done();
      });
    });
  });

}

var Promise = RSVP.Promise;
// Kudos to @Octane at https://github.com/getify/native-promise-only/issues/5 for this, and @getify for pinging me.
describe("Thenables should not be able to run code during assimilation", function () {
    specify("resolving to a thenable", function () {
        var thenCalled = false;
        var thenable = {
            then: function () {
                thenCalled = true;
            }
        };

        Promise.resolve(thenable);
        assert.strictEqual(thenCalled, false);
    });

    specify("resolving to an evil promise", function () {
        var thenCalled = false;
        var evilPromise = Promise.resolve();
        evilPromise.then = function () {
            thenCalled = true;
        };

        Promise.resolve(evilPromise);
        assert.strictEqual(thenCalled, false);
    });
});

describe("on node 0.10.x, using process.nextTick recursively shows deprecation warning. setImmediate should be used instead.", function() {
  // (node) warning: Recursive process.nextTick detected. This will break in the next version of node. Please use setImmediate for recursive deferral.

  it("should not cause 'RangeError: Maximum call stack size exceeded'", function (done) {
    var total = 1000;

    var resolved = 0;
    var nextTick = function(depth) {
      if (depth >= total) {
        return RSVP.resolve();
      }
      var d = depth + 1;
      //console.log('entered:', d);
      return new RSVP.Promise(function(resolve){
        process.nextTick(function(){
          nextTick(d).then(function(){
            resolved++;
            //console.log('resolving:', d);
            resolve();
          });
        });
      });
    };

    nextTick(0)
        .then(function(){
          //console.log('nextTick: final');
          assert.strictEqual(resolved, total);
          done();
        });
  });

});

}).call(this,require('_process'),typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"_process":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/process/browser.js","assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js"}],"/Users/stanley/code/rsvp.js/tmp/tree_merger-tmp_dest_dir-l9APRzgC.tmp/test/rsvp.js":[function(require,module,exports){
(function (process){
(function() {
    "use strict";
    function lib$rsvp$utils$$objectOrFunction(x) {
      return typeof x === 'function' || (typeof x === 'object' && x !== null);
    }

    function lib$rsvp$utils$$isFunction(x) {
      return typeof x === 'function';
    }

    function lib$rsvp$utils$$isMaybeThenable(x) {
      return typeof x === 'object' && x !== null;
    }

    var lib$rsvp$utils$$_isArray;
    if (!Array.isArray) {
      lib$rsvp$utils$$_isArray = function (x) {
        return Object.prototype.toString.call(x) === '[object Array]';
      };
    } else {
      lib$rsvp$utils$$_isArray = Array.isArray;
    }

    var lib$rsvp$utils$$isArray = lib$rsvp$utils$$_isArray;

    var lib$rsvp$utils$$now = Date.now || function() { return new Date().getTime(); };

    function lib$rsvp$utils$$F() { }

    var lib$rsvp$utils$$o_create = (Object.create || function (o) {
      if (arguments.length > 1) {
        throw new Error('Second argument not supported');
      }
      if (typeof o !== 'object') {
        throw new TypeError('Argument must be an object');
      }
      lib$rsvp$utils$$F.prototype = o;
      return new lib$rsvp$utils$$F();
    });
    function lib$rsvp$events$$indexOf(callbacks, callback) {
      for (var i=0, l=callbacks.length; i<l; i++) {
        if (callbacks[i] === callback) { return i; }
      }

      return -1;
    }

    function lib$rsvp$events$$callbacksFor(object) {
      var callbacks = object._promiseCallbacks;

      if (!callbacks) {
        callbacks = object._promiseCallbacks = {};
      }

      return callbacks;
    }

    var lib$rsvp$events$$default = {

      /**
        `RSVP.EventTarget.mixin` extends an object with EventTarget methods. For
        Example:

        ```javascript
        var object = {};

        RSVP.EventTarget.mixin(object);

        object.on('finished', function(event) {
          // handle event
        });

        object.trigger('finished', { detail: value });
        ```

        `EventTarget.mixin` also works with prototypes:

        ```javascript
        var Person = function() {};
        RSVP.EventTarget.mixin(Person.prototype);

        var yehuda = new Person();
        var tom = new Person();

        yehuda.on('poke', function(event) {
          console.log('Yehuda says OW');
        });

        tom.on('poke', function(event) {
          console.log('Tom says OW');
        });

        yehuda.trigger('poke');
        tom.trigger('poke');
        ```

        @method mixin
        @for RSVP.EventTarget
        @private
        @param {Object} object object to extend with EventTarget methods
      */
      'mixin': function(object) {
        object['on']      = this['on'];
        object['off']     = this['off'];
        object['trigger'] = this['trigger'];
        object._promiseCallbacks = undefined;
        return object;
      },

      /**
        Registers a callback to be executed when `eventName` is triggered

        ```javascript
        object.on('event', function(eventInfo){
          // handle the event
        });

        object.trigger('event');
        ```

        @method on
        @for RSVP.EventTarget
        @private
        @param {String} eventName name of the event to listen for
        @param {Function} callback function to be called when the event is triggered.
      */
      'on': function(eventName, callback) {
        var allCallbacks = lib$rsvp$events$$callbacksFor(this), callbacks;

        callbacks = allCallbacks[eventName];

        if (!callbacks) {
          callbacks = allCallbacks[eventName] = [];
        }

        if (lib$rsvp$events$$indexOf(callbacks, callback) === -1) {
          callbacks.push(callback);
        }
      },

      /**
        You can use `off` to stop firing a particular callback for an event:

        ```javascript
        function doStuff() { // do stuff! }
        object.on('stuff', doStuff);

        object.trigger('stuff'); // doStuff will be called

        // Unregister ONLY the doStuff callback
        object.off('stuff', doStuff);
        object.trigger('stuff'); // doStuff will NOT be called
        ```

        If you don't pass a `callback` argument to `off`, ALL callbacks for the
        event will not be executed when the event fires. For example:

        ```javascript
        var callback1 = function(){};
        var callback2 = function(){};

        object.on('stuff', callback1);
        object.on('stuff', callback2);

        object.trigger('stuff'); // callback1 and callback2 will be executed.

        object.off('stuff');
        object.trigger('stuff'); // callback1 and callback2 will not be executed!
        ```

        @method off
        @for RSVP.EventTarget
        @private
        @param {String} eventName event to stop listening to
        @param {Function} callback optional argument. If given, only the function
        given will be removed from the event's callback queue. If no `callback`
        argument is given, all callbacks will be removed from the event's callback
        queue.
      */
      'off': function(eventName, callback) {
        var allCallbacks = lib$rsvp$events$$callbacksFor(this), callbacks, index;

        if (!callback) {
          allCallbacks[eventName] = [];
          return;
        }

        callbacks = allCallbacks[eventName];

        index = lib$rsvp$events$$indexOf(callbacks, callback);

        if (index !== -1) { callbacks.splice(index, 1); }
      },

      /**
        Use `trigger` to fire custom events. For example:

        ```javascript
        object.on('foo', function(){
          console.log('foo event happened!');
        });
        object.trigger('foo');
        // 'foo event happened!' logged to the console
        ```

        You can also pass a value as a second argument to `trigger` that will be
        passed as an argument to all event listeners for the event:

        ```javascript
        object.on('foo', function(value){
          console.log(value.name);
        });

        object.trigger('foo', { name: 'bar' });
        // 'bar' logged to the console
        ```

        @method trigger
        @for RSVP.EventTarget
        @private
        @param {String} eventName name of the event to be triggered
        @param {Any} options optional value to be passed to any event handlers for
        the given `eventName`
      */
      'trigger': function(eventName, options) {
        var allCallbacks = lib$rsvp$events$$callbacksFor(this), callbacks, callback;

        if (callbacks = allCallbacks[eventName]) {
          // Don't cache the callbacks.length since it may grow
          for (var i=0; i<callbacks.length; i++) {
            callback = callbacks[i];

            callback(options);
          }
        }
      }
    };

    var lib$rsvp$config$$config = {
      instrument: false
    };

    lib$rsvp$events$$default['mixin'](lib$rsvp$config$$config);

    function lib$rsvp$config$$configure(name, value) {
      if (name === 'onerror') {
        // handle for legacy users that expect the actual
        // error to be passed to their function added via
        // `RSVP.configure('onerror', someFunctionHere);`
        lib$rsvp$config$$config['on']('error', value);
        return;
      }

      if (arguments.length === 2) {
        lib$rsvp$config$$config[name] = value;
      } else {
        return lib$rsvp$config$$config[name];
      }
    }

    var lib$rsvp$instrument$$queue = [];

    function lib$rsvp$instrument$$scheduleFlush() {
      setTimeout(function() {
        var entry;
        for (var i = 0; i < lib$rsvp$instrument$$queue.length; i++) {
          entry = lib$rsvp$instrument$$queue[i];

          var payload = entry.payload;

          payload.guid = payload.key + payload.id;
          payload.childGuid = payload.key + payload.childId;
          if (payload.error) {
            payload.stack = payload.error.stack;
          }

          lib$rsvp$config$$config['trigger'](entry.name, entry.payload);
        }
        lib$rsvp$instrument$$queue.length = 0;
      }, 50);
    }

    function lib$rsvp$instrument$$instrument(eventName, promise, child) {
      if (1 === lib$rsvp$instrument$$queue.push({
          name: eventName,
          payload: {
            key: promise._guidKey,
            id:  promise._id,
            eventName: eventName,
            detail: promise._result,
            childId: child && child._id,
            label: promise._label,
            timeStamp: lib$rsvp$utils$$now(),
            error: lib$rsvp$config$$config["instrument-with-stack"] ? new Error(promise._label) : null
          }})) {
            lib$rsvp$instrument$$scheduleFlush();
          }
      }
    var lib$rsvp$instrument$$default = lib$rsvp$instrument$$instrument;

    function  lib$rsvp$$internal$$withOwnPromise() {
      return new TypeError('A promises callback cannot return that same promise.');
    }

    function lib$rsvp$$internal$$noop() {}

    var lib$rsvp$$internal$$PENDING   = void 0;
    var lib$rsvp$$internal$$FULFILLED = 1;
    var lib$rsvp$$internal$$REJECTED  = 2;

    var lib$rsvp$$internal$$GET_THEN_ERROR = new lib$rsvp$$internal$$ErrorObject();

    function lib$rsvp$$internal$$getThen(promise) {
      try {
        return promise.then;
      } catch(error) {
        lib$rsvp$$internal$$GET_THEN_ERROR.error = error;
        return lib$rsvp$$internal$$GET_THEN_ERROR;
      }
    }

    function lib$rsvp$$internal$$tryThen(then, value, fulfillmentHandler, rejectionHandler) {
      try {
        then.call(value, fulfillmentHandler, rejectionHandler);
      } catch(e) {
        return e;
      }
    }

    function lib$rsvp$$internal$$handleForeignThenable(promise, thenable, then) {
      lib$rsvp$config$$config.async(function(promise) {
        var sealed = false;
        var error = lib$rsvp$$internal$$tryThen(then, thenable, function(value) {
          if (sealed) { return; }
          sealed = true;
          if (thenable !== value) {
            lib$rsvp$$internal$$resolve(promise, value);
          } else {
            lib$rsvp$$internal$$fulfill(promise, value);
          }
        }, function(reason) {
          if (sealed) { return; }
          sealed = true;

          lib$rsvp$$internal$$reject(promise, reason);
        }, 'Settle: ' + (promise._label || ' unknown promise'));

        if (!sealed && error) {
          sealed = true;
          lib$rsvp$$internal$$reject(promise, error);
        }
      }, promise);
    }

    function lib$rsvp$$internal$$handleOwnThenable(promise, thenable) {
      if (thenable._state === lib$rsvp$$internal$$FULFILLED) {
        lib$rsvp$$internal$$fulfill(promise, thenable._result);
      } else if (thenable._state === lib$rsvp$$internal$$REJECTED) {
        thenable._onError = null;
        lib$rsvp$$internal$$reject(promise, thenable._result);
      } else {
        lib$rsvp$$internal$$subscribe(thenable, undefined, function(value) {
          if (thenable !== value) {
            lib$rsvp$$internal$$resolve(promise, value);
          } else {
            lib$rsvp$$internal$$fulfill(promise, value);
          }
        }, function(reason) {
          lib$rsvp$$internal$$reject(promise, reason);
        });
      }
    }

    function lib$rsvp$$internal$$handleMaybeThenable(promise, maybeThenable) {
      if (maybeThenable.constructor === promise.constructor) {
        lib$rsvp$$internal$$handleOwnThenable(promise, maybeThenable);
      } else {
        var then = lib$rsvp$$internal$$getThen(maybeThenable);

        if (then === lib$rsvp$$internal$$GET_THEN_ERROR) {
          lib$rsvp$$internal$$reject(promise, lib$rsvp$$internal$$GET_THEN_ERROR.error);
        } else if (then === undefined) {
          lib$rsvp$$internal$$fulfill(promise, maybeThenable);
        } else if (lib$rsvp$utils$$isFunction(then)) {
          lib$rsvp$$internal$$handleForeignThenable(promise, maybeThenable, then);
        } else {
          lib$rsvp$$internal$$fulfill(promise, maybeThenable);
        }
      }
    }

    function lib$rsvp$$internal$$resolve(promise, value) {
      if (promise === value) {
        lib$rsvp$$internal$$fulfill(promise, value);
      } else if (lib$rsvp$utils$$objectOrFunction(value)) {
        lib$rsvp$$internal$$handleMaybeThenable(promise, value);
      } else {
        lib$rsvp$$internal$$fulfill(promise, value);
      }
    }

    function lib$rsvp$$internal$$publishRejection(promise) {
      if (promise._onError) {
        promise._onError(promise._result);
      }

      lib$rsvp$$internal$$publish(promise);
    }

    function lib$rsvp$$internal$$fulfill(promise, value) {
      if (promise._state !== lib$rsvp$$internal$$PENDING) { return; }

      promise._result = value;
      promise._state = lib$rsvp$$internal$$FULFILLED;

      if (promise._subscribers.length === 0) {
        if (lib$rsvp$config$$config.instrument) {
          lib$rsvp$instrument$$default('fulfilled', promise);
        }
      } else {
        lib$rsvp$config$$config.async(lib$rsvp$$internal$$publish, promise);
      }
    }

    function lib$rsvp$$internal$$reject(promise, reason) {
      if (promise._state !== lib$rsvp$$internal$$PENDING) { return; }
      promise._state = lib$rsvp$$internal$$REJECTED;
      promise._result = reason;
      lib$rsvp$config$$config.async(lib$rsvp$$internal$$publishRejection, promise);
    }

    function lib$rsvp$$internal$$subscribe(parent, child, onFulfillment, onRejection) {
      var subscribers = parent._subscribers;
      var length = subscribers.length;

      parent._onError = null;

      subscribers[length] = child;
      subscribers[length + lib$rsvp$$internal$$FULFILLED] = onFulfillment;
      subscribers[length + lib$rsvp$$internal$$REJECTED]  = onRejection;

      if (length === 0 && parent._state) {
        lib$rsvp$config$$config.async(lib$rsvp$$internal$$publish, parent);
      }
    }

    function lib$rsvp$$internal$$publish(promise) {
      var subscribers = promise._subscribers;
      var settled = promise._state;

      if (lib$rsvp$config$$config.instrument) {
        lib$rsvp$instrument$$default(settled === lib$rsvp$$internal$$FULFILLED ? 'fulfilled' : 'rejected', promise);
      }

      if (subscribers.length === 0) { return; }

      var child, callback, detail = promise._result;

      for (var i = 0; i < subscribers.length; i += 3) {
        child = subscribers[i];
        callback = subscribers[i + settled];

        if (child) {
          lib$rsvp$$internal$$invokeCallback(settled, child, callback, detail);
        } else {
          callback(detail);
        }
      }

      promise._subscribers.length = 0;
    }

    function lib$rsvp$$internal$$ErrorObject() {
      this.error = null;
    }

    var lib$rsvp$$internal$$TRY_CATCH_ERROR = new lib$rsvp$$internal$$ErrorObject();

    function lib$rsvp$$internal$$tryCatch(callback, detail) {
      try {
        return callback(detail);
      } catch(e) {
        lib$rsvp$$internal$$TRY_CATCH_ERROR.error = e;
        return lib$rsvp$$internal$$TRY_CATCH_ERROR;
      }
    }

    function lib$rsvp$$internal$$invokeCallback(settled, promise, callback, detail) {
      var hasCallback = lib$rsvp$utils$$isFunction(callback),
          value, error, succeeded, failed;

      if (hasCallback) {
        value = lib$rsvp$$internal$$tryCatch(callback, detail);

        if (value === lib$rsvp$$internal$$TRY_CATCH_ERROR) {
          failed = true;
          error = value.error;
          value = null;
        } else {
          succeeded = true;
        }

        if (promise === value) {
          lib$rsvp$$internal$$reject(promise, lib$rsvp$$internal$$withOwnPromise());
          return;
        }

      } else {
        value = detail;
        succeeded = true;
      }

      if (promise._state !== lib$rsvp$$internal$$PENDING) {
        // noop
      } else if (hasCallback && succeeded) {
        lib$rsvp$$internal$$resolve(promise, value);
      } else if (failed) {
        lib$rsvp$$internal$$reject(promise, error);
      } else if (settled === lib$rsvp$$internal$$FULFILLED) {
        lib$rsvp$$internal$$fulfill(promise, value);
      } else if (settled === lib$rsvp$$internal$$REJECTED) {
        lib$rsvp$$internal$$reject(promise, value);
      }
    }

    function lib$rsvp$$internal$$initializePromise(promise, resolver) {
      var resolved = false;
      try {
        resolver(function resolvePromise(value){
          if (resolved) { return; }
          resolved = true;
          lib$rsvp$$internal$$resolve(promise, value);
        }, function rejectPromise(reason) {
          if (resolved) { return; }
          resolved = true;
          lib$rsvp$$internal$$reject(promise, reason);
        });
      } catch(e) {
        lib$rsvp$$internal$$reject(promise, e);
      }
    }

    function lib$rsvp$enumerator$$makeSettledResult(state, position, value) {
      if (state === lib$rsvp$$internal$$FULFILLED) {
        return {
          state: 'fulfilled',
          value: value
        };
      } else {
        return {
          state: 'rejected',
          reason: value
        };
      }
    }

    function lib$rsvp$enumerator$$Enumerator(Constructor, input, abortOnReject, label) {
      this._instanceConstructor = Constructor;
      this.promise = new Constructor(lib$rsvp$$internal$$noop, label);
      this._abortOnReject = abortOnReject;

      if (this._validateInput(input)) {
        this._input     = input;
        this.length     = input.length;
        this._remaining = input.length;

        this._init();

        if (this.length === 0) {
          lib$rsvp$$internal$$fulfill(this.promise, this._result);
        } else {
          this.length = this.length || 0;
          this._enumerate();
          if (this._remaining === 0) {
            lib$rsvp$$internal$$fulfill(this.promise, this._result);
          }
        }
      } else {
        lib$rsvp$$internal$$reject(this.promise, this._validationError());
      }
    }

    var lib$rsvp$enumerator$$default = lib$rsvp$enumerator$$Enumerator;

    lib$rsvp$enumerator$$Enumerator.prototype._validateInput = function(input) {
      return lib$rsvp$utils$$isArray(input);
    };

    lib$rsvp$enumerator$$Enumerator.prototype._validationError = function() {
      return new Error('Array Methods must be provided an Array');
    };

    lib$rsvp$enumerator$$Enumerator.prototype._init = function() {
      this._result = new Array(this.length);
    };

    lib$rsvp$enumerator$$Enumerator.prototype._enumerate = function() {
      var length  = this.length;
      var promise = this.promise;
      var input   = this._input;

      for (var i = 0; promise._state === lib$rsvp$$internal$$PENDING && i < length; i++) {
        this._eachEntry(input[i], i);
      }
    };

    lib$rsvp$enumerator$$Enumerator.prototype._eachEntry = function(entry, i) {
      var c = this._instanceConstructor;
      if (lib$rsvp$utils$$isMaybeThenable(entry)) {
        if (entry.constructor === c && entry._state !== lib$rsvp$$internal$$PENDING) {
          entry._onError = null;
          this._settledAt(entry._state, i, entry._result);
        } else {
          this._willSettleAt(c.resolve(entry), i);
        }
      } else {
        this._remaining--;
        this._result[i] = this._makeResult(lib$rsvp$$internal$$FULFILLED, i, entry);
      }
    };

    lib$rsvp$enumerator$$Enumerator.prototype._settledAt = function(state, i, value) {
      var promise = this.promise;

      if (promise._state === lib$rsvp$$internal$$PENDING) {
        this._remaining--;

        if (this._abortOnReject && state === lib$rsvp$$internal$$REJECTED) {
          lib$rsvp$$internal$$reject(promise, value);
        } else {
          this._result[i] = this._makeResult(state, i, value);
        }
      }

      if (this._remaining === 0) {
        lib$rsvp$$internal$$fulfill(promise, this._result);
      }
    };

    lib$rsvp$enumerator$$Enumerator.prototype._makeResult = function(state, i, value) {
      return value;
    };

    lib$rsvp$enumerator$$Enumerator.prototype._willSettleAt = function(promise, i) {
      var enumerator = this;

      lib$rsvp$$internal$$subscribe(promise, undefined, function(value) {
        enumerator._settledAt(lib$rsvp$$internal$$FULFILLED, i, value);
      }, function(reason) {
        enumerator._settledAt(lib$rsvp$$internal$$REJECTED, i, reason);
      });
    };
    function lib$rsvp$promise$all$$all(entries, label) {
      return new lib$rsvp$enumerator$$default(this, entries, true /* abort on reject */, label).promise;
    }
    var lib$rsvp$promise$all$$default = lib$rsvp$promise$all$$all;
    function lib$rsvp$promise$race$$race(entries, label) {
      /*jshint validthis:true */
      var Constructor = this;

      var promise = new Constructor(lib$rsvp$$internal$$noop, label);

      if (!lib$rsvp$utils$$isArray(entries)) {
        lib$rsvp$$internal$$reject(promise, new TypeError('You must pass an array to race.'));
        return promise;
      }

      var length = entries.length;

      function onFulfillment(value) {
        lib$rsvp$$internal$$resolve(promise, value);
      }

      function onRejection(reason) {
        lib$rsvp$$internal$$reject(promise, reason);
      }

      for (var i = 0; promise._state === lib$rsvp$$internal$$PENDING && i < length; i++) {
        lib$rsvp$$internal$$subscribe(Constructor.resolve(entries[i]), undefined, onFulfillment, onRejection);
      }

      return promise;
    }
    var lib$rsvp$promise$race$$default = lib$rsvp$promise$race$$race;
    function lib$rsvp$promise$resolve$$resolve(object, label) {
      /*jshint validthis:true */
      var Constructor = this;

      if (object && typeof object === 'object' && object.constructor === Constructor) {
        return object;
      }

      var promise = new Constructor(lib$rsvp$$internal$$noop, label);
      lib$rsvp$$internal$$resolve(promise, object);
      return promise;
    }
    var lib$rsvp$promise$resolve$$default = lib$rsvp$promise$resolve$$resolve;
    function lib$rsvp$promise$reject$$reject(reason, label) {
      /*jshint validthis:true */
      var Constructor = this;
      var promise = new Constructor(lib$rsvp$$internal$$noop, label);
      lib$rsvp$$internal$$reject(promise, reason);
      return promise;
    }
    var lib$rsvp$promise$reject$$default = lib$rsvp$promise$reject$$reject;

    var lib$rsvp$promise$$guidKey = 'rsvp_' + lib$rsvp$utils$$now() + '-';
    var lib$rsvp$promise$$counter = 0;

    function lib$rsvp$promise$$needsResolver() {
      throw new TypeError('You must pass a resolver function as the first argument to the promise constructor');
    }

    function lib$rsvp$promise$$needsNew() {
      throw new TypeError("Failed to construct 'Promise': Please use the 'new' operator, this object constructor cannot be called as a function.");
    }

    /**
      Promise objects represent the eventual result of an asynchronous operation. The
      primary way of interacting with a promise is through its `then` method, which
      registers callbacks to receive either a promise’s eventual value or the reason
      why the promise cannot be fulfilled.

      Terminology
      -----------

      - `promise` is an object or function with a `then` method whose behavior conforms to this specification.
      - `thenable` is an object or function that defines a `then` method.
      - `value` is any legal JavaScript value (including undefined, a thenable, or a promise).
      - `exception` is a value that is thrown using the throw statement.
      - `reason` is a value that indicates why a promise was rejected.
      - `settled` the final resting state of a promise, fulfilled or rejected.

      A promise can be in one of three states: pending, fulfilled, or rejected.

      Promises that are fulfilled have a fulfillment value and are in the fulfilled
      state.  Promises that are rejected have a rejection reason and are in the
      rejected state.  A fulfillment value is never a thenable.

      Promises can also be said to *resolve* a value.  If this value is also a
      promise, then the original promise's settled state will match the value's
      settled state.  So a promise that *resolves* a promise that rejects will
      itself reject, and a promise that *resolves* a promise that fulfills will
      itself fulfill.


      Basic Usage:
      ------------

      ```js
      var promise = new Promise(function(resolve, reject) {
        // on success
        resolve(value);

        // on failure
        reject(reason);
      });

      promise.then(function(value) {
        // on fulfillment
      }, function(reason) {
        // on rejection
      });
      ```

      Advanced Usage:
      ---------------

      Promises shine when abstracting away asynchronous interactions such as
      `XMLHttpRequest`s.

      ```js
      function getJSON(url) {
        return new Promise(function(resolve, reject){
          var xhr = new XMLHttpRequest();

          xhr.open('GET', url);
          xhr.onreadystatechange = handler;
          xhr.responseType = 'json';
          xhr.setRequestHeader('Accept', 'application/json');
          xhr.send();

          function handler() {
            if (this.readyState === this.DONE) {
              if (this.status === 200) {
                resolve(this.response);
              } else {
                reject(new Error('getJSON: `' + url + '` failed with status: [' + this.status + ']'));
              }
            }
          };
        });
      }

      getJSON('/posts.json').then(function(json) {
        // on fulfillment
      }, function(reason) {
        // on rejection
      });
      ```

      Unlike callbacks, promises are great composable primitives.

      ```js
      Promise.all([
        getJSON('/posts'),
        getJSON('/comments')
      ]).then(function(values){
        values[0] // => postsJSON
        values[1] // => commentsJSON

        return values;
      });
      ```

      @class RSVP.Promise
      @param {function} resolver
      @param {String} label optional string for labeling the promise.
      Useful for tooling.
      @constructor
    */
    function lib$rsvp$promise$$Promise(resolver, label) {
      this._id = lib$rsvp$promise$$counter++;
      this._label = label;
      this._state = undefined;
      this._result = undefined;
      this._subscribers = [];

      if (lib$rsvp$config$$config.instrument) {
        lib$rsvp$instrument$$default('created', this);
      }

      if (lib$rsvp$$internal$$noop !== resolver) {
        if (!lib$rsvp$utils$$isFunction(resolver)) {
          lib$rsvp$promise$$needsResolver();
        }

        if (!(this instanceof lib$rsvp$promise$$Promise)) {
          lib$rsvp$promise$$needsNew();
        }

        lib$rsvp$$internal$$initializePromise(this, resolver);
      }
    }

    var lib$rsvp$promise$$default = lib$rsvp$promise$$Promise;

    // deprecated
    lib$rsvp$promise$$Promise.cast = lib$rsvp$promise$resolve$$default;
    lib$rsvp$promise$$Promise.all = lib$rsvp$promise$all$$default;
    lib$rsvp$promise$$Promise.race = lib$rsvp$promise$race$$default;
    lib$rsvp$promise$$Promise.resolve = lib$rsvp$promise$resolve$$default;
    lib$rsvp$promise$$Promise.reject = lib$rsvp$promise$reject$$default;

    lib$rsvp$promise$$Promise.prototype = {
      constructor: lib$rsvp$promise$$Promise,

      _guidKey: lib$rsvp$promise$$guidKey,

      _onError: function (reason) {
        lib$rsvp$config$$config.async(function(promise) {
          setTimeout(function() {
            if (promise._onError) {
              lib$rsvp$config$$config['trigger']('error', reason);
            }
          }, 0);
        }, this);
      },

    /**
      The primary way of interacting with a promise is through its `then` method,
      which registers callbacks to receive either a promise's eventual value or the
      reason why the promise cannot be fulfilled.

      ```js
      findUser().then(function(user){
        // user is available
      }, function(reason){
        // user is unavailable, and you are given the reason why
      });
      ```

      Chaining
      --------

      The return value of `then` is itself a promise.  This second, 'downstream'
      promise is resolved with the return value of the first promise's fulfillment
      or rejection handler, or rejected if the handler throws an exception.

      ```js
      findUser().then(function (user) {
        return user.name;
      }, function (reason) {
        return 'default name';
      }).then(function (userName) {
        // If `findUser` fulfilled, `userName` will be the user's name, otherwise it
        // will be `'default name'`
      });

      findUser().then(function (user) {
        throw new Error('Found user, but still unhappy');
      }, function (reason) {
        throw new Error('`findUser` rejected and we're unhappy');
      }).then(function (value) {
        // never reached
      }, function (reason) {
        // if `findUser` fulfilled, `reason` will be 'Found user, but still unhappy'.
        // If `findUser` rejected, `reason` will be '`findUser` rejected and we're unhappy'.
      });
      ```
      If the downstream promise does not specify a rejection handler, rejection reasons will be propagated further downstream.

      ```js
      findUser().then(function (user) {
        throw new PedagogicalException('Upstream error');
      }).then(function (value) {
        // never reached
      }).then(function (value) {
        // never reached
      }, function (reason) {
        // The `PedgagocialException` is propagated all the way down to here
      });
      ```

      Assimilation
      ------------

      Sometimes the value you want to propagate to a downstream promise can only be
      retrieved asynchronously. This can be achieved by returning a promise in the
      fulfillment or rejection handler. The downstream promise will then be pending
      until the returned promise is settled. This is called *assimilation*.

      ```js
      findUser().then(function (user) {
        return findCommentsByAuthor(user);
      }).then(function (comments) {
        // The user's comments are now available
      });
      ```

      If the assimliated promise rejects, then the downstream promise will also reject.

      ```js
      findUser().then(function (user) {
        return findCommentsByAuthor(user);
      }).then(function (comments) {
        // If `findCommentsByAuthor` fulfills, we'll have the value here
      }, function (reason) {
        // If `findCommentsByAuthor` rejects, we'll have the reason here
      });
      ```

      Simple Example
      --------------

      Synchronous Example

      ```javascript
      var result;

      try {
        result = findResult();
        // success
      } catch(reason) {
        // failure
      }
      ```

      Errback Example

      ```js
      findResult(function(result, err){
        if (err) {
          // failure
        } else {
          // success
        }
      });
      ```

      Promise Example;

      ```javascript
      findResult().then(function(result){
        // success
      }, function(reason){
        // failure
      });
      ```

      Advanced Example
      --------------

      Synchronous Example

      ```javascript
      var author, books;

      try {
        author = findAuthor();
        books  = findBooksByAuthor(author);
        // success
      } catch(reason) {
        // failure
      }
      ```

      Errback Example

      ```js

      function foundBooks(books) {

      }

      function failure(reason) {

      }

      findAuthor(function(author, err){
        if (err) {
          failure(err);
          // failure
        } else {
          try {
            findBoooksByAuthor(author, function(books, err) {
              if (err) {
                failure(err);
              } else {
                try {
                  foundBooks(books);
                } catch(reason) {
                  failure(reason);
                }
              }
            });
          } catch(error) {
            failure(err);
          }
          // success
        }
      });
      ```

      Promise Example;

      ```javascript
      findAuthor().
        then(findBooksByAuthor).
        then(function(books){
          // found books
      }).catch(function(reason){
        // something went wrong
      });
      ```

      @method then
      @param {Function} onFulfilled
      @param {Function} onRejected
      @param {String} label optional string for labeling the promise.
      Useful for tooling.
      @return {Promise}
    */
      then: function(onFulfillment, onRejection, label) {
        var parent = this;
        var state = parent._state;

        if (state === lib$rsvp$$internal$$FULFILLED && !onFulfillment || state === lib$rsvp$$internal$$REJECTED && !onRejection) {
          if (lib$rsvp$config$$config.instrument) {
            lib$rsvp$instrument$$default('chained', this, this);
          }
          return this;
        }

        parent._onError = null;

        var child = new this.constructor(lib$rsvp$$internal$$noop, label);
        var result = parent._result;

        if (lib$rsvp$config$$config.instrument) {
          lib$rsvp$instrument$$default('chained', parent, child);
        }

        if (state) {
          var callback = arguments[state - 1];
          lib$rsvp$config$$config.async(function(){
            lib$rsvp$$internal$$invokeCallback(state, child, callback, result);
          });
        } else {
          lib$rsvp$$internal$$subscribe(parent, child, onFulfillment, onRejection);
        }

        return child;
      },

    /**
      `catch` is simply sugar for `then(undefined, onRejection)` which makes it the same
      as the catch block of a try/catch statement.

      ```js
      function findAuthor(){
        throw new Error('couldn't find that author');
      }

      // synchronous
      try {
        findAuthor();
      } catch(reason) {
        // something went wrong
      }

      // async with promises
      findAuthor().catch(function(reason){
        // something went wrong
      });
      ```

      @method catch
      @param {Function} onRejection
      @param {String} label optional string for labeling the promise.
      Useful for tooling.
      @return {Promise}
    */
      'catch': function(onRejection, label) {
        return this.then(null, onRejection, label);
      },

    /**
      `finally` will be invoked regardless of the promise's fate just as native
      try/catch/finally behaves

      Synchronous example:

      ```js
      findAuthor() {
        if (Math.random() > 0.5) {
          throw new Error();
        }
        return new Author();
      }

      try {
        return findAuthor(); // succeed or fail
      } catch(error) {
        return findOtherAuther();
      } finally {
        // always runs
        // doesn't affect the return value
      }
      ```

      Asynchronous example:

      ```js
      findAuthor().catch(function(reason){
        return findOtherAuther();
      }).finally(function(){
        // author was either found, or not
      });
      ```

      @method finally
      @param {Function} callback
      @param {String} label optional string for labeling the promise.
      Useful for tooling.
      @return {Promise}
    */
      'finally': function(callback, label) {
        var constructor = this.constructor;

        return this.then(function(value) {
          return constructor.resolve(callback()).then(function(){
            return value;
          });
        }, function(reason) {
          return constructor.resolve(callback()).then(function(){
            throw reason;
          });
        }, label);
      }
    };

    function lib$rsvp$all$settled$$AllSettled(Constructor, entries, label) {
      this._superConstructor(Constructor, entries, false /* don't abort on reject */, label);
    }

    lib$rsvp$all$settled$$AllSettled.prototype = lib$rsvp$utils$$o_create(lib$rsvp$enumerator$$default.prototype);
    lib$rsvp$all$settled$$AllSettled.prototype._superConstructor = lib$rsvp$enumerator$$default;
    lib$rsvp$all$settled$$AllSettled.prototype._makeResult = lib$rsvp$enumerator$$makeSettledResult;
    lib$rsvp$all$settled$$AllSettled.prototype._validationError = function() {
      return new Error('allSettled must be called with an array');
    };

    function lib$rsvp$all$settled$$allSettled(entries, label) {
      return new lib$rsvp$all$settled$$AllSettled(lib$rsvp$promise$$default, entries, label).promise;
    }
    var lib$rsvp$all$settled$$default = lib$rsvp$all$settled$$allSettled;
    function lib$rsvp$all$$all(array, label) {
      return lib$rsvp$promise$$default.all(array, label);
    }
    var lib$rsvp$all$$default = lib$rsvp$all$$all;
    var lib$rsvp$asap$$len = 0;
    var lib$rsvp$asap$$toString = {}.toString;
    var lib$rsvp$asap$$vertxNext;
    function lib$rsvp$asap$$asap(callback, arg) {
      lib$rsvp$asap$$queue[lib$rsvp$asap$$len] = callback;
      lib$rsvp$asap$$queue[lib$rsvp$asap$$len + 1] = arg;
      lib$rsvp$asap$$len += 2;
      if (lib$rsvp$asap$$len === 2) {
        // If len is 1, that means that we need to schedule an async flush.
        // If additional callbacks are queued before the queue is flushed, they
        // will be processed by this flush that we are scheduling.
        lib$rsvp$asap$$scheduleFlush();
      }
    }

    var lib$rsvp$asap$$default = lib$rsvp$asap$$asap;

    var lib$rsvp$asap$$browserWindow = (typeof window !== 'undefined') ? window : undefined;
    var lib$rsvp$asap$$browserGlobal = lib$rsvp$asap$$browserWindow || {};
    var lib$rsvp$asap$$BrowserMutationObserver = lib$rsvp$asap$$browserGlobal.MutationObserver || lib$rsvp$asap$$browserGlobal.WebKitMutationObserver;
    var lib$rsvp$asap$$isNode = typeof process !== 'undefined' && {}.toString.call(process) === '[object process]';

    // test for web worker but not in IE10
    var lib$rsvp$asap$$isWorker = typeof Uint8ClampedArray !== 'undefined' &&
      typeof importScripts !== 'undefined' &&
      typeof MessageChannel !== 'undefined';

    // node
    function lib$rsvp$asap$$useNextTick() {
      var nextTick = process.nextTick;
      // node version 0.10.x displays a deprecation warning when nextTick is used recursively
      // setImmediate should be used instead instead
      var version = process.versions.node.match(/^(?:(\d+)\.)?(?:(\d+)\.)?(\*|\d+)$/);
      if (Array.isArray(version) && version[1] === '0' && version[2] === '10') {
        nextTick = setImmediate;
      }
      return function() {
        nextTick(lib$rsvp$asap$$flush);
      };
    }

    // vertx
    function lib$rsvp$asap$$useVertxTimer() {
      return function() {
        lib$rsvp$asap$$vertxNext(lib$rsvp$asap$$flush);
      };
    }

    function lib$rsvp$asap$$useMutationObserver() {
      var iterations = 0;
      var observer = new lib$rsvp$asap$$BrowserMutationObserver(lib$rsvp$asap$$flush);
      var node = document.createTextNode('');
      observer.observe(node, { characterData: true });

      return function() {
        node.data = (iterations = ++iterations % 2);
      };
    }

    // web worker
    function lib$rsvp$asap$$useMessageChannel() {
      var channel = new MessageChannel();
      channel.port1.onmessage = lib$rsvp$asap$$flush;
      return function () {
        channel.port2.postMessage(0);
      };
    }

    function lib$rsvp$asap$$useSetTimeout() {
      return function() {
        setTimeout(lib$rsvp$asap$$flush, 1);
      };
    }

    var lib$rsvp$asap$$queue = new Array(1000);
    function lib$rsvp$asap$$flush() {
      for (var i = 0; i < lib$rsvp$asap$$len; i+=2) {
        var callback = lib$rsvp$asap$$queue[i];
        var arg = lib$rsvp$asap$$queue[i+1];

        callback(arg);

        lib$rsvp$asap$$queue[i] = undefined;
        lib$rsvp$asap$$queue[i+1] = undefined;
      }

      lib$rsvp$asap$$len = 0;
    }

    function lib$rsvp$asap$$attemptVertex() {
      try {
        var r = require;
        var vertx = r('vertx');
        lib$rsvp$asap$$vertxNext = vertx.runOnLoop || vertx.runOnContext;
        return lib$rsvp$asap$$useVertxTimer();
      } catch(e) {
        return lib$rsvp$asap$$useSetTimeout();
      }
    }

    var lib$rsvp$asap$$scheduleFlush;
    // Decide what async method to use to triggering processing of queued callbacks:
    if (lib$rsvp$asap$$isNode) {
      lib$rsvp$asap$$scheduleFlush = lib$rsvp$asap$$useNextTick();
    } else if (lib$rsvp$asap$$BrowserMutationObserver) {
      lib$rsvp$asap$$scheduleFlush = lib$rsvp$asap$$useMutationObserver();
    } else if (lib$rsvp$asap$$isWorker) {
      lib$rsvp$asap$$scheduleFlush = lib$rsvp$asap$$useMessageChannel();
    } else if (lib$rsvp$asap$$browserWindow === undefined && typeof require === 'function') {
      lib$rsvp$asap$$scheduleFlush = lib$rsvp$asap$$attemptVertex();
    } else {
      lib$rsvp$asap$$scheduleFlush = lib$rsvp$asap$$useSetTimeout();
    }
    function lib$rsvp$defer$$defer(label) {
      var deferred = { };

      deferred['promise'] = new lib$rsvp$promise$$default(function(resolve, reject) {
        deferred['resolve'] = resolve;
        deferred['reject'] = reject;
      }, label);

      return deferred;
    }
    var lib$rsvp$defer$$default = lib$rsvp$defer$$defer;
    function lib$rsvp$filter$$filter(promises, filterFn, label) {
      return lib$rsvp$promise$$default.all(promises, label).then(function(values) {
        if (!lib$rsvp$utils$$isFunction(filterFn)) {
          throw new TypeError("You must pass a function as filter's second argument.");
        }

        var length = values.length;
        var filtered = new Array(length);

        for (var i = 0; i < length; i++) {
          filtered[i] = filterFn(values[i]);
        }

        return lib$rsvp$promise$$default.all(filtered, label).then(function(filtered) {
          var results = new Array(length);
          var newLength = 0;

          for (var i = 0; i < length; i++) {
            if (filtered[i]) {
              results[newLength] = values[i];
              newLength++;
            }
          }

          results.length = newLength;

          return results;
        });
      });
    }
    var lib$rsvp$filter$$default = lib$rsvp$filter$$filter;

    function lib$rsvp$promise$hash$$PromiseHash(Constructor, object, label) {
      this._superConstructor(Constructor, object, true, label);
    }

    var lib$rsvp$promise$hash$$default = lib$rsvp$promise$hash$$PromiseHash;

    lib$rsvp$promise$hash$$PromiseHash.prototype = lib$rsvp$utils$$o_create(lib$rsvp$enumerator$$default.prototype);
    lib$rsvp$promise$hash$$PromiseHash.prototype._superConstructor = lib$rsvp$enumerator$$default;
    lib$rsvp$promise$hash$$PromiseHash.prototype._init = function() {
      this._result = {};
    };

    lib$rsvp$promise$hash$$PromiseHash.prototype._validateInput = function(input) {
      return input && typeof input === 'object';
    };

    lib$rsvp$promise$hash$$PromiseHash.prototype._validationError = function() {
      return new Error('Promise.hash must be called with an object');
    };

    lib$rsvp$promise$hash$$PromiseHash.prototype._enumerate = function() {
      var promise = this.promise;
      var input   = this._input;
      var results = [];

      for (var key in input) {
        if (promise._state === lib$rsvp$$internal$$PENDING && input.hasOwnProperty(key)) {
          results.push({
            position: key,
            entry: input[key]
          });
        }
      }

      var length = results.length;
      this._remaining = length;
      var result;

      for (var i = 0; promise._state === lib$rsvp$$internal$$PENDING && i < length; i++) {
        result = results[i];
        this._eachEntry(result.entry, result.position);
      }
    };

    function lib$rsvp$hash$settled$$HashSettled(Constructor, object, label) {
      this._superConstructor(Constructor, object, false, label);
    }

    lib$rsvp$hash$settled$$HashSettled.prototype = lib$rsvp$utils$$o_create(lib$rsvp$promise$hash$$default.prototype);
    lib$rsvp$hash$settled$$HashSettled.prototype._superConstructor = lib$rsvp$enumerator$$default;
    lib$rsvp$hash$settled$$HashSettled.prototype._makeResult = lib$rsvp$enumerator$$makeSettledResult;

    lib$rsvp$hash$settled$$HashSettled.prototype._validationError = function() {
      return new Error('hashSettled must be called with an object');
    };

    function lib$rsvp$hash$settled$$hashSettled(object, label) {
      return new lib$rsvp$hash$settled$$HashSettled(lib$rsvp$promise$$default, object, label).promise;
    }
    var lib$rsvp$hash$settled$$default = lib$rsvp$hash$settled$$hashSettled;
    function lib$rsvp$hash$$hash(object, label) {
      return new lib$rsvp$promise$hash$$default(lib$rsvp$promise$$default, object, label).promise;
    }
    var lib$rsvp$hash$$default = lib$rsvp$hash$$hash;
    function lib$rsvp$map$$map(promises, mapFn, label) {
      return lib$rsvp$promise$$default.all(promises, label).then(function(values) {
        if (!lib$rsvp$utils$$isFunction(mapFn)) {
          throw new TypeError("You must pass a function as map's second argument.");
        }

        var length = values.length;
        var results = new Array(length);

        for (var i = 0; i < length; i++) {
          results[i] = mapFn(values[i]);
        }

        return lib$rsvp$promise$$default.all(results, label);
      });
    }
    var lib$rsvp$map$$default = lib$rsvp$map$$map;

    function lib$rsvp$node$$Result() {
      this.value = undefined;
    }

    var lib$rsvp$node$$ERROR = new lib$rsvp$node$$Result();
    var lib$rsvp$node$$GET_THEN_ERROR = new lib$rsvp$node$$Result();

    function lib$rsvp$node$$getThen(obj) {
      try {
       return obj.then;
      } catch(error) {
        lib$rsvp$node$$ERROR.value= error;
        return lib$rsvp$node$$ERROR;
      }
    }


    function lib$rsvp$node$$tryApply(f, s, a) {
      try {
        f.apply(s, a);
      } catch(error) {
        lib$rsvp$node$$ERROR.value = error;
        return lib$rsvp$node$$ERROR;
      }
    }

    function lib$rsvp$node$$makeObject(_, argumentNames) {
      var obj = {};
      var name;
      var i;
      var length = _.length;
      var args = new Array(length);

      for (var x = 0; x < length; x++) {
        args[x] = _[x];
      }

      for (i = 0; i < argumentNames.length; i++) {
        name = argumentNames[i];
        obj[name] = args[i + 1];
      }

      return obj;
    }

    function lib$rsvp$node$$arrayResult(_) {
      var length = _.length;
      var args = new Array(length - 1);

      for (var i = 1; i < length; i++) {
        args[i - 1] = _[i];
      }

      return args;
    }

    function lib$rsvp$node$$wrapThenable(then, promise) {
      return {
        then: function(onFulFillment, onRejection) {
          return then.call(promise, onFulFillment, onRejection);
        }
      };
    }

    function lib$rsvp$node$$denodeify(nodeFunc, options) {
      var fn = function() {
        var self = this;
        var l = arguments.length;
        var args = new Array(l + 1);
        var arg;
        var promiseInput = false;

        for (var i = 0; i < l; ++i) {
          arg = arguments[i];

          if (!promiseInput) {
            // TODO: clean this up
            promiseInput = lib$rsvp$node$$needsPromiseInput(arg);
            if (promiseInput === lib$rsvp$node$$GET_THEN_ERROR) {
              var p = new lib$rsvp$promise$$default(lib$rsvp$$internal$$noop);
              lib$rsvp$$internal$$reject(p, lib$rsvp$node$$GET_THEN_ERROR.value);
              return p;
            } else if (promiseInput && promiseInput !== true) {
              arg = lib$rsvp$node$$wrapThenable(promiseInput, arg);
            }
          }
          args[i] = arg;
        }

        var promise = new lib$rsvp$promise$$default(lib$rsvp$$internal$$noop);

        args[l] = function(err, val) {
          if (err)
            lib$rsvp$$internal$$reject(promise, err);
          else if (options === undefined)
            lib$rsvp$$internal$$resolve(promise, val);
          else if (options === true)
            lib$rsvp$$internal$$resolve(promise, lib$rsvp$node$$arrayResult(arguments));
          else if (lib$rsvp$utils$$isArray(options))
            lib$rsvp$$internal$$resolve(promise, lib$rsvp$node$$makeObject(arguments, options));
          else
            lib$rsvp$$internal$$resolve(promise, val);
        };

        if (promiseInput) {
          return lib$rsvp$node$$handlePromiseInput(promise, args, nodeFunc, self);
        } else {
          return lib$rsvp$node$$handleValueInput(promise, args, nodeFunc, self);
        }
      };

      fn.__proto__ = nodeFunc;

      return fn;
    }

    var lib$rsvp$node$$default = lib$rsvp$node$$denodeify;

    function lib$rsvp$node$$handleValueInput(promise, args, nodeFunc, self) {
      var result = lib$rsvp$node$$tryApply(nodeFunc, self, args);
      if (result === lib$rsvp$node$$ERROR) {
        lib$rsvp$$internal$$reject(promise, result.value);
      }
      return promise;
    }

    function lib$rsvp$node$$handlePromiseInput(promise, args, nodeFunc, self){
      return lib$rsvp$promise$$default.all(args).then(function(args){
        var result = lib$rsvp$node$$tryApply(nodeFunc, self, args);
        if (result === lib$rsvp$node$$ERROR) {
          lib$rsvp$$internal$$reject(promise, result.value);
        }
        return promise;
      });
    }

    function lib$rsvp$node$$needsPromiseInput(arg) {
      if (arg && typeof arg === 'object') {
        if (arg.constructor === lib$rsvp$promise$$default) {
          return true;
        } else {
          return lib$rsvp$node$$getThen(arg);
        }
      } else {
        return false;
      }
    }
    function lib$rsvp$race$$race(array, label) {
      return lib$rsvp$promise$$default.race(array, label);
    }
    var lib$rsvp$race$$default = lib$rsvp$race$$race;
    function lib$rsvp$reject$$reject(reason, label) {
      return lib$rsvp$promise$$default.reject(reason, label);
    }
    var lib$rsvp$reject$$default = lib$rsvp$reject$$reject;
    function lib$rsvp$resolve$$resolve(value, label) {
      return lib$rsvp$promise$$default.resolve(value, label);
    }
    var lib$rsvp$resolve$$default = lib$rsvp$resolve$$resolve;
    function lib$rsvp$rethrow$$rethrow(reason) {
      setTimeout(function() {
        throw reason;
      });
      throw reason;
    }
    var lib$rsvp$rethrow$$default = lib$rsvp$rethrow$$rethrow;

    // default async is asap;
    lib$rsvp$config$$config.async = lib$rsvp$asap$$default;
    var lib$rsvp$$cast = lib$rsvp$resolve$$default;
    function lib$rsvp$$async(callback, arg) {
      lib$rsvp$config$$config.async(callback, arg);
    }

    function lib$rsvp$$on() {
      lib$rsvp$config$$config['on'].apply(lib$rsvp$config$$config, arguments);
    }

    function lib$rsvp$$off() {
      lib$rsvp$config$$config['off'].apply(lib$rsvp$config$$config, arguments);
    }

    // Set up instrumentation through `window.__PROMISE_INTRUMENTATION__`
    if (typeof window !== 'undefined' && typeof window['__PROMISE_INSTRUMENTATION__'] === 'object') {
      var lib$rsvp$$callbacks = window['__PROMISE_INSTRUMENTATION__'];
      lib$rsvp$config$$configure('instrument', true);
      for (var lib$rsvp$$eventName in lib$rsvp$$callbacks) {
        if (lib$rsvp$$callbacks.hasOwnProperty(lib$rsvp$$eventName)) {
          lib$rsvp$$on(lib$rsvp$$eventName, lib$rsvp$$callbacks[lib$rsvp$$eventName]);
        }
      }
    }

    var lib$rsvp$umd$$RSVP = {
      'race': lib$rsvp$race$$default,
      'Promise': lib$rsvp$promise$$default,
      'allSettled': lib$rsvp$all$settled$$default,
      'hash': lib$rsvp$hash$$default,
      'hashSettled': lib$rsvp$hash$settled$$default,
      'denodeify': lib$rsvp$node$$default,
      'on': lib$rsvp$$on,
      'off': lib$rsvp$$off,
      'map': lib$rsvp$map$$default,
      'filter': lib$rsvp$filter$$default,
      'resolve': lib$rsvp$resolve$$default,
      'reject': lib$rsvp$reject$$default,
      'all': lib$rsvp$all$$default,
      'rethrow': lib$rsvp$rethrow$$default,
      'defer': lib$rsvp$defer$$default,
      'EventTarget': lib$rsvp$events$$default,
      'configure': lib$rsvp$config$$configure,
      'async': lib$rsvp$$async
    };

    /* global define:true module:true window: true */
    if (typeof define === 'function' && define['amd']) {
      define(function() { return lib$rsvp$umd$$RSVP; });
    } else if (typeof module !== 'undefined' && module['exports']) {
      module['exports'] = lib$rsvp$umd$$RSVP;
    } else if (typeof this !== 'undefined') {
      this['RSVP'] = lib$rsvp$umd$$RSVP;
    }
}).call(this);

//# sourceMappingURL=rsvp.js.map
}).call(this,require('_process'))
},{"_process":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/process/browser.js"}],"/Users/stanley/code/rsvp.js/tmp/tree_merger-tmp_dest_dir-l9APRzgC.tmp/test/test-adapter.js":[function(require,module,exports){
(function (global){
/*global RSVP*/

var assert = require('assert');
var g = typeof window !== 'undefined' ?
               window : typeof global !== 'undefined' ? global : this;

var RSVP = g.RSVP || require('./rsvp.js');
var defer = RSVP.defer;
var resolve = RSVP.resolve;
var reject = RSVP.reject;

module.exports = g.adapter = {
  resolved: resolve,
  rejected: reject,
  deferred: defer,
  RSVP: RSVP
};

}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./rsvp.js":"/Users/stanley/code/rsvp.js/tmp/tree_merger-tmp_dest_dir-l9APRzgC.tmp/test/rsvp.js","assert":"/Users/stanley/code/rsvp.js/node_modules/broccoli-watchify/node_modules/browserify/node_modules/assert/assert.js"}]},{},["./test/index.js"]);
