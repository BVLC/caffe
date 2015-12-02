'use strict';
/*jshint asi: true */

var test            =  require('tap').test;
var convert         =  require('convert-source-map');
var commentRegex    =  require('convert-source-map').commentRegex;
var combine         =  require('..');
var mappingsFromMap =  require('../lib/mappings-from-map');

function checkMappings(foo, sm, lineOffset) {
    function inspect(obj, depth) {
        return require('util').inspect(obj, false, depth || 5, true);
    }

    var fooMappings = mappingsFromMap(foo);
    var mappings = mappingsFromMap(sm);

    var genLinesOffset = true;
    var origLinesSame = true;
    for (var i = 0; i < mappings.length; i++) {
        var fooGen = fooMappings[i].generated;
        var fooOrig = fooMappings[i].original;
        var gen = mappings[i].generated
        var orig = mappings[i].original;

        if (gen.column !== fooGen.column || gen.line !== (fooGen.line + lineOffset)) {
          console.error(
            'generated mapping at %s not offset properly:\ninput:  [%s]\noutput:[%s]\n\n',
            i ,
            inspect(fooGen),
            inspect(gen)
          );
          genLinesOffset = false;
        }

        if (orig.column !== fooOrig.column || orig.line !== fooOrig.line) {
          console.error(
            'original mapping at %s is not the same as the genrated mapping:\ninput:  [%s]\noutput:[%s]\n\n',
            i ,
            inspect(fooOrig),
            inspect(orig)
          );
          origLinesSame = false;
        }
    }
    return { genLinesOffset: genLinesOffset, origLinesSame: origLinesSame };
}

var foo = {
  version        :  3,
  file           :  'foo.js',
  sourceRoot     :  '',
  sources        :  [ 'foo.coffee' ],
  names          :  [],
  mappings       :  ';AAAA;CAAA;CAAA,CAAA,CAAA,IAAO,GAAK;CAAZ',
  sourcesContent :  [ 'console.log(require \'./bar.js\')\n' ] };

test('add one file with inlined source', function (t) {

  var mapComment = convert.fromObject(foo).toComment();
  var file = {
      id: 'xyz'
    , source: '(function() {\n\n  console.log(require(\'./bar.js\'));\n\n}).call(this);\n' + '\n' + mapComment
    , sourceFile: 'foo.js'
  };

  var lineOffset = 3
  var base64 = combine.create()
    .addFile(file, { line: lineOffset })
    .base64()

  var sm = convert.fromBase64(base64).toObject();
  var res = checkMappings(foo, sm, lineOffset);

  t.ok(res.genLinesOffset, 'all generated lines are offset properly and columns unchanged')
  t.ok(res.origLinesSame, 'all original lines and columns are unchanged')
  t.equal(sm.sourcesContent[0], foo.sourcesContent[0], 'includes the original source')
  t.equal(sm.sources[0], 'foo.coffee', 'includes original filename')
  t.end()
});


test('add one file without inlined source', function (t) {

  var mapComment = convert
    .fromObject(foo)
    .setProperty('sourcesContent', [])
    .toComment();

  var file = {
      id: 'xyz'
    , source: '(function() {\n\n  console.log(require(\'./bar.js\'));\n\n}).call(this);\n' + '\n' + mapComment
    , sourceFile: 'foo.js'
  };

  var lineOffset = 3
  var base64 = combine.create()
    .addFile(file, { line: lineOffset })
    .base64()

  var sm = convert.fromBase64(base64).toObject();
  var mappings = mappingsFromMap(sm);

  t.equal(sm.sourcesContent[0], file.source, 'includes the generated source')
  t.equal(sm.sources[0], 'foo.js', 'includes generated filename')

  t.deepEqual(
      mappings
    , [ { generated: { line: 4, column: 0 },
        original: { line: 1, column: 0 },
        source: 'foo.js', name: undefined },
      { generated: { line: 5, column: 0 },
        original: { line: 2, column: 0 },
        source: 'foo.js', name: undefined },
      { generated: { line: 6, column: 0 },
        original: { line: 3, column: 0 },
        source: 'foo.js', name: undefined },
      { generated: { line: 7, column: 0 },
        original: { line: 4, column: 0 },
        source: 'foo.js', name: undefined },
      { generated: { line: 8, column: 0 },
        original: { line: 5, column: 0 },
        source: 'foo.js', name: undefined },
      { generated: { line: 9, column: 0 },
        original: { line: 6, column: 0 },
        source: 'foo.js', name: undefined },
      { generated: { line: 10, column: 0 },
        original: { line: 7, column: 0 },
        source: 'foo.js', name: undefined } ]
    , 'generates mappings offset by the given line'
  )
  t.end()
})

test('remove comments', function (t) {
  var mapComment = convert.fromObject(foo).toComment();

  function sourcemapComments(src) {
    var matches = src.match(commentRegex);
    return matches ? matches.length : 0;
  }

  t.equal(sourcemapComments('var a = 1;\n' + mapComment), 1);

  [ ''
  , 'var a = 1;\n' + mapComment
  , 'var a = 1;\n' + mapComment + '\nvar b = 5;\n' + mapComment
  ] .forEach(function (x) {
    var removed = combine.removeComments(x)
    t.equal(sourcemapComments(removed), 0)
  })
  t.end()
})
