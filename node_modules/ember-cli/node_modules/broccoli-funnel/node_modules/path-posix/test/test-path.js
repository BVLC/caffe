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

var assert = require('assert');

var path = require('..');

var f = '/foo/test/simple/test-path.js';

assert.equal(path.basename(f), 'test-path.js');
assert.equal(path.basename(f, '.js'), 'test-path');
assert.equal(path.basename(''), '');
assert.equal(path.basename('/dir/basename.ext'), 'basename.ext');
assert.equal(path.basename('/basename.ext'), 'basename.ext');
assert.equal(path.basename('basename.ext'), 'basename.ext');
assert.equal(path.basename('basename.ext/'), 'basename.ext');
assert.equal(path.basename('basename.ext//'), 'basename.ext');

// On unix a backslash is just treated as any other character.
assert.equal(path.basename('\\dir\\basename.ext'), '\\dir\\basename.ext');
assert.equal(path.basename('\\basename.ext'), '\\basename.ext');
assert.equal(path.basename('basename.ext'), 'basename.ext');
assert.equal(path.basename('basename.ext\\'), 'basename.ext\\');
assert.equal(path.basename('basename.ext\\\\'), 'basename.ext\\\\');

// POSIX filenames may include control characters
// c.f. http://www.dwheeler.com/essays/fixing-unix-linux-filenames.html
  var controlCharFilename = 'Icon' + String.fromCharCode(13);
  assert.equal(path.basename('/a/b/' + controlCharFilename),
               controlCharFilename);

assert.equal(path.extname(f), '.js');

assert.equal(path.dirname(f).substr(-11), 'test/simple');
assert.equal(path.dirname('/a/b/'), '/a');
assert.equal(path.dirname('/a/b'), '/a');
assert.equal(path.dirname('/a'), '/');
assert.equal(path.dirname(''), '.');
assert.equal(path.dirname('/'), '/');
assert.equal(path.dirname('////'), '/');


assert.equal(path.extname(''), '');
assert.equal(path.extname('/path/to/file'), '');
assert.equal(path.extname('/path/to/file.ext'), '.ext');
assert.equal(path.extname('/path.to/file.ext'), '.ext');
assert.equal(path.extname('/path.to/file'), '');
assert.equal(path.extname('/path.to/.file'), '');
assert.equal(path.extname('/path.to/.file.ext'), '.ext');
assert.equal(path.extname('/path/to/f.ext'), '.ext');
assert.equal(path.extname('/path/to/..ext'), '.ext');
assert.equal(path.extname('file'), '');
assert.equal(path.extname('file.ext'), '.ext');
assert.equal(path.extname('.file'), '');
assert.equal(path.extname('.file.ext'), '.ext');
assert.equal(path.extname('/file'), '');
assert.equal(path.extname('/file.ext'), '.ext');
assert.equal(path.extname('/.file'), '');
assert.equal(path.extname('/.file.ext'), '.ext');
assert.equal(path.extname('.path/file.ext'), '.ext');
assert.equal(path.extname('file.ext.ext'), '.ext');
assert.equal(path.extname('file.'), '.');
assert.equal(path.extname('.'), '');
assert.equal(path.extname('./'), '');
assert.equal(path.extname('.file.ext'), '.ext');
assert.equal(path.extname('.file'), '');
assert.equal(path.extname('.file.'), '.');
assert.equal(path.extname('.file..'), '.');
assert.equal(path.extname('..'), '');
assert.equal(path.extname('../'), '');
assert.equal(path.extname('..file.ext'), '.ext');
assert.equal(path.extname('..file'), '.file');
assert.equal(path.extname('..file.'), '.');
assert.equal(path.extname('..file..'), '.');
assert.equal(path.extname('...'), '.');
assert.equal(path.extname('...ext'), '.ext');
assert.equal(path.extname('....'), '.');
assert.equal(path.extname('file.ext/'), '.ext');
assert.equal(path.extname('file.ext//'), '.ext');
assert.equal(path.extname('file/'), '');
assert.equal(path.extname('file//'), '');
assert.equal(path.extname('file./'), '.');
assert.equal(path.extname('file.//'), '.');

// On unix, backspace is a valid name component like any other character.
assert.equal(path.extname('.\\'), '');
assert.equal(path.extname('..\\'), '.\\');
assert.equal(path.extname('file.ext\\'), '.ext\\');
assert.equal(path.extname('file.ext\\\\'), '.ext\\\\');
assert.equal(path.extname('file\\'), '');
assert.equal(path.extname('file\\\\'), '');
assert.equal(path.extname('file.\\'), '.\\');
assert.equal(path.extname('file.\\\\'), '.\\\\');

// path.join tests
var failures = [];
var joinTests =
    // arguments                     result
    [[['.', 'x/b', '..', '/b/c.js'], 'x/b/c.js'],
     [['/.', 'x/b', '..', '/b/c.js'], '/x/b/c.js'],
     [['/foo', '../../../bar'], '/bar'],
     [['foo', '../../../bar'], '../../bar'],
     [['foo/', '../../../bar'], '../../bar'],
     [['foo/x', '../../../bar'], '../bar'],
     [['foo/x', './bar'], 'foo/x/bar'],
     [['foo/x/', './bar'], 'foo/x/bar'],
     [['foo/x/', '.', 'bar'], 'foo/x/bar'],
     [['./'], './'],
     [['.', './'], './'],
     [['.', '.', '.'], '.'],
     [['.', './', '.'], '.'],
     [['.', '/./', '.'], '.'],
     [['.', '/////./', '.'], '.'],
     [['.'], '.'],
     [['', '.'], '.'],
     [['', 'foo'], 'foo'],
     [['foo', '/bar'], 'foo/bar'],
     [['', '/foo'], '/foo'],
     [['', '', '/foo'], '/foo'],
     [['', '', 'foo'], 'foo'],
     [['foo', ''], 'foo'],
     [['foo/', ''], 'foo/'],
     [['foo', '', '/bar'], 'foo/bar'],
     [['./', '..', '/foo'], '../foo'],
     [['./', '..', '..', '/foo'], '../../foo'],
     [['.', '..', '..', '/foo'], '../../foo'],
     [['', '..', '..', '/foo'], '../../foo'],
     [['/'], '/'],
     [['/', '.'], '/'],
     [['/', '..'], '/'],
     [['/', '..', '..'], '/'],
     [[''], '.'],
     [['', ''], '.'],
     [[' /foo'], ' /foo'],
     [[' ', 'foo'], ' /foo'],
     [[' ', '.'], ' '],
     [[' ', '/'], ' /'],
     [[' ', ''], ' '],
     [['/', 'foo'], '/foo'],
     [['/', '/foo'], '/foo'],
     [['/', '//foo'], '/foo'],
     [['/', '', '/foo'], '/foo'],
     [['', '/', 'foo'], '/foo'],
     [['', '/', '/foo'], '/foo']
    ];

// Run the join tests.
joinTests.forEach(function(test) {
  var actual = path.join.apply(path, test[0]);
  var expected = test[1];
  var message = 'path.join(' + test[0].map(JSON.stringify).join(',') + ')' +
                '\n  expect=' + JSON.stringify(expected) +
                '\n  actual=' + JSON.stringify(actual);
  if (actual !== expected) failures.push('\n' + message);
  // assert.equal(actual, expected, message);
});
assert.equal(failures.length, 0, failures.join(''));
var joinThrowTests = [true, false, 7, null, {}, undefined, [], NaN];
joinThrowTests.forEach(function(test) {
  assert.throws(function() {
    path.join(test);
  }, TypeError);
  assert.throws(function() {
    path.resolve(test);
  }, TypeError);
});


// path normalize tests

assert.equal(path.normalize('./fixtures///b/../b/c.js'),
             'fixtures/b/c.js');
assert.equal(path.normalize('/foo/../../../bar'), '/bar');
assert.equal(path.normalize('a//b//../b'), 'a/b');
assert.equal(path.normalize('a//b//./c'), 'a/b/c');
assert.equal(path.normalize('a//b//.'), 'a/b');

// path.resolve tests
  var resolveTests =
      // arguments                                    result
      [[['/var/lib', '../', 'file/'], '/var/file'],
       [['/var/lib', '/../', 'file/'], '/file'],
       [['a/b/c/', '../../..'], process.cwd()],
       [['.'], process.cwd()],
       [['/some/dir', '.', '/absolute/'], '/absolute']];

var failures = [];
resolveTests.forEach(function(test) {
  var actual = path.resolve.apply(path, test[0]);
  var expected = test[1];
  var message = 'path.resolve(' + test[0].map(JSON.stringify).join(',') + ')' +
                '\n  expect=' + JSON.stringify(expected) +
                '\n  actual=' + JSON.stringify(actual);
  if (actual !== expected) failures.push('\n' + message);
  // assert.equal(actual, expected, message);
});
assert.equal(failures.length, 0, failures.join(''));

// path.isAbsolute tests
assert.equal(path.isAbsolute('/home/foo'), true);
assert.equal(path.isAbsolute('/home/foo/..'), true);
assert.equal(path.isAbsolute('bar/'), false);
assert.equal(path.isAbsolute('./baz'), false);

// path.relative tests
  var relativeTests =
      // arguments                    result
      [['/var/lib', '/var', '..'],
       ['/var/lib', '/bin', '../../bin'],
       ['/var/lib', '/var/lib', ''],
       ['/var/lib', '/var/apache', '../apache'],
       ['/var/', '/var/lib', 'lib'],
       ['/', '/var/lib', 'var/lib']];
var failures = [];
relativeTests.forEach(function(test) {
  var actual = path.relative(test[0], test[1]);
  var expected = test[2];
  var message = 'path.relative(' +
                test.slice(0, 2).map(JSON.stringify).join(',') +
                ')' +
                '\n  expect=' + JSON.stringify(expected) +
                '\n  actual=' + JSON.stringify(actual);
  if (actual !== expected) failures.push('\n' + message);
});
assert.equal(failures.length, 0, failures.join(''));

assert.equal(path.sep, '/');

// path.delimiter tests
assert.equal(path.delimiter, ':');
