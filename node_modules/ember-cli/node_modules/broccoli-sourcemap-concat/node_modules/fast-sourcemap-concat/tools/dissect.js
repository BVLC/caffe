#!/usr/bin/env node

var srcURL = require('source-map-url');
var SourceMap = require('../lib/source-map');

if (process.argv.length < 3) {
  process.stderr.write("Usage: dissect.js <javascript_file> [<sourcemap>]\n");
  process.exit(-1);
}

var fs = require('fs');
var path = require('path');
var src = fs.readFileSync(process.argv[2], 'utf-8');
var map;

if (srcURL.existsIn(src)) {
  var url = srcURL.getFrom(src);
  src = srcURL.removeFrom(src);
  map = SourceMap.prototype._resolveSourcemap(process.argv[2], url);
} else {
  map = JSON.parse(fs.readFileSync(process.argv[3], 'utf-8'));
}
var Coder = require('../lib/coder');
var colWidth = 60;
var newline = /\n\r?/;

var lines = src.split(newline);
var mappings = map.mappings.split(';');
var splitContents;
if (map.sourcesContent) {
  splitContents = map.sourcesContent.map(function(src){return src ? src.split(newline) : [];});
} else {
  splitContents = map.sources.map(function(name){
    return fs.readFileSync(path.join(path.dirname(process.argv[2]), name), 'utf-8').split(newline);
  });
}
var decoder = new Coder();

function padUpTo(str, padding) {
  var extra = padding - str.length;
  while (extra > 0) {
    extra--;
    str += ' ';
  }
  if (str.length > padding) {
    str = str.slice(0, padding-3) + '...';
  }
  return str;
}

var lastOrigLine;
var value;
function decode(mapping) {
  value = decoder.decode(mapping);
}

for (var i=0; i<lines.length;i++) {
  if (i >= mappings.length) {
    console.log("Ran out of mappings");
    process.exit(-1);
  }

  decoder.resetColumn();
  mappings[i].split(',').forEach(decode);

  var differential = mappings[i].split(',').map(function(elt) {
    return decoder.debug(elt);
  }).join('|');

  var fileDesc ='';
  var origLine = '';

  if (value.hasOwnProperty('source')) {
    fileDesc += map.sources[value.source];
  }
  if (value.hasOwnProperty('originalLine')) {
    fileDesc += ':' + value.originalLine;
    var whichSource = splitContents[value.source];
    if (whichSource) {
      origLine = whichSource[value.originalLine];
    }
    if (typeof origLine === 'undefined'){
      origLine = '<BAD LINE REFERENCE: ' + value.source + ',' + value.originalLine + '>';
    }
  }

  console.log([
    padUpTo(differential, colWidth),
    padUpTo(fileDesc, colWidth),
    padUpTo(origLine === lastOrigLine ? '' : origLine, colWidth),
    ' | ',
    padUpTo(lines[i], colWidth)
  ].join(''));
  lastOrigLine = origLine;
  if (i % 20 === 0) {
    var sep = '';
    for (var col=0; col<colWidth*4;col++){
      sep += '-';
    }
    console.log(sep);
  }
}
