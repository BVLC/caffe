var fs = require('fs');
var srcURL = require('source-map-url');
var path = require('path');
var RSVP = require('rsvp');
var mkdirp = require('mkdirp');
var Coder = require('./coder');
var crypto = require('crypto');
var chalk = require('chalk');
var debug = require('debug');

module.exports = SourceMap;
function SourceMap(opts) {
  if (!(this instanceof SourceMap)) {
    return new SourceMap(opts);
  }
  if (!opts || !opts.outputFile) {
    throw new Error("Must specify outputFile");
  }
  if (opts.mapFile && !opts.mapURL) {
    throw new Error("must specify the mapURL when setting a custom mapFile");
  }
  this.debug = debug('fast-sourcemap-concat:');
  this.baseDir = opts.baseDir;
  this.outputFile = opts.outputFile;
  this.cache = opts.cache;
  this.mapFile = opts.mapFile || this.outputFile.replace(/\.js$/, '') + '.map';
  this.mapURL = opts.mapURL || path.basename(this.mapFile);
  this.mapCommentType = opts.mapCommentType || 'line';
  this._initializeStream();

  this.content = {
    version: 3,
    sources: [],
    sourcesContent: [],
    names: [],
    mappings: ''
  };
  if (opts.sourceRoot) {
    this.content.sourceRoot = opts.sourceRoot;
  }
  this.content.file = opts.file || path.basename(opts.outputFile);
  this.encoder = new Coder();

  // Keep track of what column we're currently outputing in the
  // concatenated sourcecode file. Notice that we don't track line
  // though -- line is implicit in this.content.mappings.
  this.column = 0;

  // Keep track of how many lines worth of mappings we've output into
  // the concatenated sourcemap. We use this to correct broken input
  // sourcemaps that don't match the length of their sourcecode.
  this.linesMapped = 0;
}
SourceMap.prototype._resolveFile = function(filename) {
  if (this.baseDir && filename.slice(0,1) !== '/') {
    filename = path.join(this.baseDir, filename);
  }
  return filename;
};

SourceMap.prototype._initializeStream = function() {
  mkdirp.sync(path.dirname(this.outputFile));
  this.stream = fs.createWriteStream(this.outputFile);
};


SourceMap.prototype.addFile = function(filename) {
  var source = fs.readFileSync(this._resolveFile(filename), 'utf-8');
  return this.addFileSource(filename, source);
};

SourceMap.prototype.addFileSource = function(filename, source) {
  var url;
  var inputSrcMap;

  if (source.length === 0) {
    return;
  }

  if (srcURL.existsIn(source)) {
    url = srcURL.getFrom(source);
    source = srcURL.removeFrom(source);
  }

  if (this.content.mappings.length > 0 && !/[;,]$/.test(this.content.mappings)) {
    this.content.mappings += ',';
  }


  if (url && (inputSrcMap = this._resolveSourcemap(filename, url))) {
    source = this._addMap(filename, inputSrcMap, source);
  } else {
    this.debug('generating new map: ' + filename);
    this.content.sources.push(filename);
    this.content.sourcesContent.push(source);
    this._generateNewMap(source);
  }

  this.stream.write(source);
};

SourceMap.prototype._cacheEncoderResults = function(key, operations, filename) {
  var encoderState = this.encoder.copy();
  var initialLinesMapped = this.linesMapped;
  var cacheEntry = this.cache[key];
  var finalState;

  if (cacheEntry) {
    // The cache contains the encoder-differential for our file. So
    // this updates encoderState to the final value our encoder will
    // have after processing the file.
    encoderState.decode(cacheEntry.encoder);
    // We provide that final value as a performance hint.
    operations.call(this, {
      encoder: encoderState,
      lines: cacheEntry.lines
    });

    this.debug('encoder cache hit: ' + filename);

  } else {

    // Run the operations with no hint because we don't have one yet.
    operations.call(this);
    // Then store the encoder differential in the cache.
    finalState = this.encoder.copy();
    finalState.subtract(encoderState);
    this.cache[key] = {
      encoder: finalState.serialize(),
      lines: this.linesMapped - initialLinesMapped
    };

    this.debug('encoder cache prime: ' + filename);
  }
};

// This is useful for things like separators that you're appending to
// your JS file that don't need to have their own source mapping, but
// will alter the line numbering for subsequent files.
SourceMap.prototype.addSpace = function(source) {
  this.stream.write(source);
  var lineCount = countNewLines(source);
  if (lineCount === 0) {
    this.column += source.length;
  } else {
    this.column = 0;
    var mappings = this.content.mappings;
    for (var i = 0; i < lineCount; i++) {
      mappings += ';';
    }
    this.content.mappings = mappings;
  }
};

SourceMap.prototype._generateNewMap = function(source) {
  var mappings = this.content.mappings;
  var lineCount = countNewLines(source);

  mappings += this.encoder.encode({
    generatedColumn: this.column,
    source: this.content.sources.length-1,
    originalLine: 0,
    originalColumn: 0
  });

  if (lineCount === 0) {
    // no newline in the source. Keep outputting one big line.
    this.column += source.length;
  } else {
    // end the line
    this.column = 0;
    this.encoder.resetColumn();
    mappings += ';';
    this.encoder.adjustLine(lineCount-1);
  }

  // For the remainder of the lines (if any), we're just following
  // one-to-one.
  for (var i = 0; i < lineCount-1; i++) {
    mappings += 'AACA;';
  }
  this.linesMapped += lineCount;
  this.content.mappings = mappings;
};

SourceMap.prototype._resolveSourcemap = function(filename, url) {
  var srcMap;
  var match = /^data:.+?;base64,/.exec(url);

  try {
    if (match) {
      srcMap = new Buffer(url.slice(match[0].length), 'base64');
    } else if (this.baseDir && url.slice(0,1) === '/') {
      srcMap = fs.readFileSync(
        path.join(this.baseDir, url),
        'utf8'
      );
    } else {
      srcMap = fs.readFileSync(
        path.join(path.dirname(this._resolveFile(filename)), url),
        'utf8'
      );
    }
    return JSON.parse(srcMap);
  } catch (err) {
    this._warn('Warning: ignoring input sourcemap for ' + filename + ' because ' + err.message);
  }
};


SourceMap.prototype._addMap = function(filename, srcMap, source) {
  var initialLinesMapped = this.linesMapped;
  var haveLines = countNewLines(source);
  var self = this;

  if (this.cache) {
    this._cacheEncoderResults(hash(source), function(cacheHint) {
      self._assimilateExistingMap(filename, srcMap, cacheHint);
    }, filename);
  } else {
    this._assimilateExistingMap(filename, srcMap);
  }

  while (this.linesMapped - initialLinesMapped < haveLines) {
    // This cleans up after upstream sourcemaps that are too short
    // for their sourcecode so they don't break the rest of our
    // mapping. Coffeescript does this.
    this.content.mappings += ';';
    this.linesMapped++;
  }
  while (haveLines < this.linesMapped - initialLinesMapped) {
    // Likewise, this cleans up after upstream sourcemaps that are
    // too long for their sourcecode.
    source += "\n";
    haveLines++;
  }
  return source;
};


SourceMap.prototype._assimilateExistingMap = function(filename, srcMap, cacheHint) {
  var content = this.content;
  var sourcesOffset = content.sources.length;
  var namesOffset = content.names.length;

  content.sources = content.sources.concat(this._resolveSources(srcMap));
  content.sourcesContent = content.sourcesContent.concat(this._resolveSourcesContent(srcMap, filename));
  while (content.sourcesContent.length > content.sources.length) {
    content.sourcesContent.pop();
  }
  while (content.sourcesContent.length < content.sources.length) {
    content.sourcesContent.push(null);
  }
  content.names = content.names.concat(srcMap.names);
  this._scanMappings(srcMap, sourcesOffset, namesOffset, cacheHint);
};

SourceMap.prototype._resolveSources = function(srcMap) {
  var baseDir = this.baseDir;
  if (!baseDir) {
    return srcMap.sources;
  }
  return srcMap.sources.map(function(src) {
    return src.replace(baseDir, '');
  });
};

SourceMap.prototype._resolveSourcesContent = function(srcMap, filename) {
  if (srcMap.sourcesContent) {
    // Upstream srcmap already had inline content, so easy.
    return srcMap.sourcesContent;
  } else {
    // Look for original sources relative to our input source filename.
    return srcMap.sources.map(function(source){
      var fullPath;
      if (source.slice(0, 1) === '/') {
        fullPath = source;
      } else {
        fullPath = path.join(path.dirname(this._resolveFile(filename)), source);
      }
      return fs.readFileSync(fullPath, 'utf-8');
    }.bind(this));
  }
};


SourceMap.prototype._scanMappings = function(srcMap, sourcesOffset, namesOffset, cacheHint) {
  var mappings = this.content.mappings;
  var decoder = new Coder();
  var inputMappings = srcMap.mappings;
  var pattern = /^([;,]*)([^;,]*)/;
  var continuation = /^([;,]*)((?:AACA;)+)/;
  var initialMappedLines = this.linesMapped;
  var match;
  var lines;

  while (inputMappings.length > 0) {
    match = pattern.exec(inputMappings);

    // If the entry was preceded by separators, copy them through.
    if (match[1]) {
      mappings += match[1];
      lines = match[1].replace(/,/g, '').length;
      if (lines > 0) {
        this.linesMapped += lines;
        this.encoder.resetColumn();
        decoder.resetColumn();
      }
    }

    // Re-encode the entry.
    if (match[2]){
      var value = decoder.decode(match[2]);
      value.generatedColumn += this.column;
      this.column = 0;
      if (sourcesOffset && value.hasOwnProperty('source')) {
        value.source += sourcesOffset;
        decoder.prev_source += sourcesOffset;
        sourcesOffset = 0;
      }
      if (namesOffset && value.hasOwnProperty('name')) {
        value.name += namesOffset;
        decoder.prev_name += namesOffset;
        namesOffset = 0;
      }
      mappings += this.encoder.encode(value);
    }

    inputMappings = inputMappings.slice(match[0].length);

    // Once we've applied any offsets, we can try to jump ahead.
    if (!sourcesOffset && !namesOffset) {
      if (cacheHint) {
        // Our cacheHint tells us what our final encoder state will be
        // after processing this file. And since we've got nothing
        // left ahead that needs rewriting, we can just copy the
        // remaining mappings over and jump to the final encoder
        // state.
        mappings += inputMappings;
        inputMappings = '';
        this.linesMapped = initialMappedLines + cacheHint.lines;
        this.encoder = cacheHint.encoder;
      }


      if ((match = continuation.exec(inputMappings))) {
        // This is a significant optimization, especially when we're
        // doing simple line-for-line concatenations.
        lines = match[2].length / 5;
        this.encoder.adjustLine(lines);
        this.encoder.resetColumn();
        decoder.adjustLine(lines);
        decoder.resetColumn();
        this.linesMapped += lines + match[1].replace(/,/g, '').length;
        mappings += match[0];
        inputMappings = inputMappings.slice(match[0].length);
      }
    }

  }
  this.content.mappings = mappings;
};

SourceMap.prototype.end = function() {
  if (this.mapCommentType === 'line') {
    this.stream.write('//# sourceMappingURL=' + this.mapURL);
  } else {
    this.stream.write('/*# sourceMappingURL=' + this.mapURL + ' */');
  }
  mkdirp.sync(path.dirname(this.mapFile));
  fs.writeFileSync(this.mapFile, JSON.stringify(this.content));
  return new RSVP.Promise(function(resolve, reject) {
    this.stream.on('finish', resolve);
    this.stream.on('error', reject);
    this.stream.end();
  }.bind(this));
};

SourceMap.prototype._warn = function(msg) {
  console.log(chalk.yellow(msg));
};

function countNewLines(src) {
  var newlinePattern = /(\r?\n)/g;
  var count = 0;
  while (newlinePattern.exec(src)) {
    count++;
  }
  return count;
}


function hash(string) {
  return crypto.createHash('md5').update(string).digest('hex');
}
