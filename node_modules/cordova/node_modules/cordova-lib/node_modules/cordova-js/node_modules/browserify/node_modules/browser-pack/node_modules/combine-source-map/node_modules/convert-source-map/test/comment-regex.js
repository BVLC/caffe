'use strict';
/*jshint asi: true */

var test = require('tap').test
  , generator = require('inline-source-map')
  , rx = require('..').commentRegex
  , mapFileRx = require('..').mapFileCommentRegex

function comment(s) {
  rx.lastIndex = 0;
  return rx.test(s + 'sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiIiwic291cmNlcyI6WyJmdW5jdGlvbiBmb28oKSB7XG4gY29uc29sZS5sb2coXCJoZWxsbyBJIGFtIGZvb1wiKTtcbiBjb25zb2xlLmxvZyhcIndobyBhcmUgeW91XCIpO1xufVxuXG5mb28oKTtcbiJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQSJ9')
}

test('comment regex old spec - @', function (t) {
  [ '//@ '
  , '  //@ '
  , '\t//@ '
  ].forEach(function (x) { t.ok(comment(x), 'matches ' + x) });

  [ '///@ ' 
  , '}}//@ '
  , ' @// @'
  ].forEach(function (x) { t.ok(!comment(x), 'does not match ' + x) })
  t.end()
})

test('comment regex new spec - #', function (t) {
  [ '//# '
  , '  //# '
  , '\t//# '
  ].forEach(function (x) { t.ok(comment(x), 'matches ' + x) });

  [ '///# ' 
  , '}}//# '
  , ' #// #'
  ].forEach(function (x) { t.ok(!comment(x), 'does not match ' + x) })
  t.end()
})

function mapFileComment(s) {
  mapFileRx.lastIndex = 0;
  return mapFileRx.test(s + 'sourceMappingURL=foo.js.map')
}

test('mapFileComment regex old spec - @', function (t) {
  [ '//@ '
  , '  //@ '
  , '\t//@ '
  ].forEach(function (x) { t.ok(mapFileComment(x), 'matches ' + x) });

  [ '///@ ' 
  , '}}//@ '
  , ' @// @'
  ].forEach(function (x) { t.ok(!mapFileComment(x), 'does not match ' + x) })
  t.end()
})

test('mapFileComment regex new spec - #', function (t) {
  [ '//# '
  , '  //# '
  , '\t//# '
  ].forEach(function (x) { t.ok(mapFileComment(x), 'matches ' + x) });

  [ '///# ' 
  , '}}//# '
  , ' #// #'
  ].forEach(function (x) { t.ok(!mapFileComment(x), 'does not match ' + x) })
  t.end()
})

function mapFileCommentWrap(s1, s2) {
  mapFileRx.lastIndex = 0;
  return mapFileRx.test(s1 + 'sourceMappingURL=foo.js.map' + s2)
}

test('mapFileComment regex /* */ old spec - @', function (t) {
  [ [ '/*@ ', '*/' ]
  , ['  /*@ ', '  */ ' ]
  , [ '\t/*@ ', ' \t*/\t ']
  ].forEach(function (x) { t.ok(mapFileCommentWrap(x[0], x[1]), 'matches ' + x.join(' :: ')) });

  [ [ '/*/*@ ', '*/' ]
  , ['}}/*@ ', '  */ ' ]
  , [ ' @/*@ ', ' \t*/\t ']
  ].forEach(function (x) { t.ok(!mapFileCommentWrap(x[0], x[1]), 'does not match ' + x.join(' :: ')) });
  t.end()
})

test('mapFileComment regex /* */ new spec - #', function (t) {
  [ [ '/*# ', '*/' ]
  , ['  /*# ', '  */ ' ]
  , [ '\t/*# ', ' \t*/\t ']
  ].forEach(function (x) { t.ok(mapFileCommentWrap(x[0], x[1]), 'matches ' + x.join(' :: ')) });

  [ [ '/*/*# ', '*/' ]
  , ['}}/*# ', '  */ ' ]
  , [ ' #/*# ', ' \t*/\t ']
  ].forEach(function (x) { t.ok(!mapFileCommentWrap(x[0], x[1]), 'does not match ' + x.join(' :: ')) });
  t.end()
})
