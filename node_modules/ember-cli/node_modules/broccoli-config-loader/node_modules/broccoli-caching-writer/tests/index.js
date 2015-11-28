'use strict';

var fs = require('fs');
var path = require('path');
var chai = require('chai'), expect = chai.expect;
var chaiAsPromised = require('chai-as-promised');
chai.use(chaiAsPromised);
var broccoli = require('broccoli');
var CachingWriter = require('..');

var builder, cachingWriter, buildCount;

function setupCachingWriter(inputNodes, options, buildCallback) {
  if (!buildCallback) buildCallback = function() { };
  buildCount = 0;
  cachingWriter = new CachingWriter(inputNodes, options);
  cachingWriter.build = function() {
    buildCount++;
    return buildCallback.call(this);
  };
  builder = new broccoli.Builder(cachingWriter);
}

function build(expectRebuild) {
  return builder.build()
    .then(function(hash) {
      return hash.directory;
    });
}

function expectRebuild() {
  var oldBuildCount = buildCount;
  return build()
    .then(function(outputPath) { // cannot use finally - expect failure would override previous error
      expect(buildCount).to.equal(oldBuildCount + 1,
        'expected rebuild to be triggered');
      return outputPath;
    });
}

function expectNoRebuild() {
  var oldBuildCount = buildCount;
  return build()
    .then(function(outputPath) { // cannot use finally - expect failure would override previous error
      expect(buildCount).to.equal(oldBuildCount,
        'expected rebuild not to be triggered');
      return outputPath;
    });
}

describe('broccoli-caching-writer', function() {
  var sourcePath = 'tests/fixtures/sample-project';
  var secondaryPath = 'tests/fixtures/other-tree';

  var existingJSFile = sourcePath + '/core.js';
  var dummyChangedFile = sourcePath + '/dummy-changed-file.txt';
  var dummyJSChangedFile = sourcePath + '/dummy-changed-file.js';

  afterEach(function() {
    if (fs.existsSync(dummyChangedFile)) {
      fs.unlinkSync(dummyChangedFile);
    }

    if (fs.existsSync(dummyJSChangedFile)) {
      fs.unlinkSync(dummyJSChangedFile);
    }

    if (builder) {
      return builder.cleanup();
    }
  });

  describe('cache invalidation', function() {
    it('calls build once at the beginning, and again only when input is changed', function() {
      setupCachingWriter([sourcePath, secondaryPath], {});

      return expectRebuild()
        .then(expectNoRebuild)
        .then(expectNoRebuild)
        .then(function() { fs.writeFileSync(dummyChangedFile, 'bergh'); })
        .then(expectRebuild)
        .then(expectNoRebuild)
        .then(function() { fs.writeFileSync(secondaryPath + '/foo-baz.js', 'bergh'); })
        .then(expectRebuild)
        .then(expectNoRebuild);
    });

    it('calls build again if existing file is changed', function() {
      setupCachingWriter([sourcePath], {});

      return expectRebuild()
        .then(function() { fs.writeFileSync(existingJSFile, '"YIPPIE"\n"KI-YAY!"\n'); })
        .then(expectRebuild)
        .then(expectNoRebuild)
        .finally(function() { fs.writeFileSync(existingJSFile, '"YIPPIE"\n'); });
    });

    it('builds once with no input nodes', function() {
      setupCachingWriter([], {});

      return expectRebuild()
        .then(expectNoRebuild());
    });

    it('does not call build again if input is changed but filtered from cache (via cacheExclude)', function() {
      setupCachingWriter([sourcePath], {
        cacheExclude: [/.*\.txt$/]
      });

      return expectRebuild()
        .then(function() { fs.writeFileSync(dummyChangedFile, 'bergh'); })
        .then(expectNoRebuild);
    });

    it('does not call updateCache again if input is changed but filtered from cache (via cacheInclude)', function() {
      setupCachingWriter([sourcePath], {
        cacheInclude: [/.*\.js$/]
      });

      return expectRebuild()
        .then(function() { fs.writeFileSync(dummyChangedFile, 'bergh'); })
        .then(expectNoRebuild);
    });

    it('does call build again if input is changed is included in the cache filter', function() {
      setupCachingWriter([sourcePath], {
        cacheInclude: [/.*\.js$/]
      });

      return expectRebuild()
        .then(function() { fs.writeFileSync(dummyJSChangedFile, 'bergh'); })
        .then(expectRebuild);
    });

    it('when inputFiles is given, calls updateCache only when any of those files are changed', function() {
      setupCachingWriter([sourcePath], {
        inputFiles: ['core.js'] // existingJSFile
      });

      return expectRebuild()
        .then(function() { fs.writeFileSync(dummyChangedFile, 'bergh'); })
        .then(expectNoRebuild)
        .then(function() { fs.writeFileSync(existingJSFile, '"YIPPIE"\n"KI-YAY!"\n'); })
        .then(expectRebuild)
        .finally(function() { fs.writeFileSync(existingJSFile, '"YIPPIE"\n'); });
    });
  });

  describe('build', function() {
    it('can read from inputPaths', function() {
      setupCachingWriter([sourcePath, secondaryPath], {}, function() {
        expect(fs.readFileSync(this.inputPaths[0] + '/core.js', {
          encoding: 'utf8'
        })).to.contain('"YIPPIE"');
        expect(fs.readFileSync(this.inputPaths[1] + '/bar.js', {
          encoding: 'utf8'
        })).to.contain('"BLAMMO!"');
      });

      return expectRebuild();
    });

    it('can write to outputPath', function() {
      setupCachingWriter([sourcePath], {}, function() {
        fs.writeFileSync(this.outputPath + '/something-cool.js', 'zomg blammo', {encoding: 'utf8'});
      });

      return expectRebuild()
        .then(function(outputPath) {
          expect(fs.readFileSync(outputPath + '/something-cool.js', {
            encoding: 'utf8'
          })).to.equal('zomg blammo');
        });
    });

    it('throws an error if not overriden', function(){
      setupCachingWriter([sourcePath], {});
      delete cachingWriter.build;

      return expect(build()).to.be.rejectedWith(/Plugin subclasses must implement/);
    });

    it('can return a promise that is resolved', function(){
      var thenCalled = false;
      setupCachingWriter([sourcePath], {}, function() {
        return {then: function(callback) {
          thenCalled = true;
          callback();
        }};
      });

      return expectRebuild().then(function() {
        expect(thenCalled).to.be.ok;
      });
    });
  });

  describe('constructor', function() {
    it('throws exception when no input nodes are provided', function() {
      expect(function() {
        new CachingWriter();
      }).to.throw(/Expected an array/);
    });

    it('throws exception when something other than an array is passed', function() {
      expect(function() {
        new CachingWriter('not/an/array');
      }).to.throw(/Expected an array/);
    });
  });

  describe('persistentOutput behavior mimics broccoli-plugin', function() {
    function buildOnce() {
      /*jshint validthis:true */
      if (!this.builtOnce) {
        this.builtOnce = true;
        fs.writeFileSync(path.join(this.outputPath, 'foo.txt'), 'yay');
      }
    }

    function isEmptied() {
      return expectRebuild()
        .then(function() { fs.writeFileSync(dummyChangedFile, 'force rebuild'); })
        .then(expectRebuild)
        .then(function(outputPath) {
          return !fs.existsSync(path.join(outputPath, 'foo.txt'));
        });
    }

    it('empties the output directory by default', function() {
      setupCachingWriter([sourcePath], {}, buildOnce);
      return expect(isEmptied()).to.be.eventually.true;
    });

    it('does not empty the output directory if persistentOutput is true', function() {
      setupCachingWriter([sourcePath], { persistentOutput: true }, buildOnce);
      return expect(isEmptied()).to.be.eventually.false;
    });
  });

  describe('shouldBeIgnored', function() {
    it('returns true if the path is included in an exclude filter', function() {
      var node = new CachingWriter([sourcePath], {
        cacheExclude: [ /.foo$/, /.bar$/ ]
      });

      expect(node.shouldBeIgnored('blah/blah/blah.foo')).to.be.ok;
      expect(node.shouldBeIgnored('blah/blah/blah.bar')).to.be.ok;
      expect(node.shouldBeIgnored('blah/blah/blah.baz')).to.not.be.ok;
    });

    it('returns false if the path is included in an include filter', function() {
      var node = new CachingWriter([sourcePath], {
        cacheInclude: [ /.foo$/, /.bar$/ ]
      });

      expect(node.shouldBeIgnored('blah/blah/blah.foo')).to.not.be.ok;
      expect(node.shouldBeIgnored('blah/blah/blah.bar')).to.not.be.ok;
    });

    it('returns true if the path is not included in an include filter', function() {
      var node = new CachingWriter([sourcePath], {
        cacheInclude: [ /.foo$/, /.bar$/ ]
      });

      expect(node.shouldBeIgnored('blah/blah/blah.baz')).to.be.ok;
    });

    it('returns false if no patterns were used', function() {
      var node = new CachingWriter([sourcePath], {});

      expect(node.shouldBeIgnored('blah/blah/blah.baz')).to.not.be.ok;
    });

    it('uses a cache to ensure we do not recalculate the filtering on subsequent attempts', function() {
      var node = new CachingWriter([sourcePath], {});

      expect(node.shouldBeIgnored('blah/blah/blah.baz')).to.not.be.ok;

      // changing the filter mid-run should have no result on
      // previously calculated paths
      node._cacheInclude = [ /.foo$/, /.bar$/ ];

      expect(node.shouldBeIgnored('blah/blah/blah.baz')).to.not.be.ok;
    });
  });

  describe('listEntries', function() {
    var listFiles;

    function getListFilesFor(options) {
      setupCachingWriter([sourcePath], options, function() {
        var writer = this;
        listFiles = this.listFiles().map(function(p) {
          return path.relative(writer.inputPaths[0], p);
        });
      });
      return expectRebuild().then(function() {
        return listFiles;
      });
    }

    it('returns an array of files keyed', function() {
      return expect(getListFilesFor({})).to.eventually.deep.equal(['core.js', 'main.js']);
    });

    it('returns an array of files keyed including only those in the include filter', function() {
      return expect(getListFilesFor({
        cacheInclude: [ /core\.js$/ ]
      })).to.eventually.deep.equal(['core.js']);
    });

    it('returns an array of files keyed ignoring those in the exclude filter', function() {
      return expect(getListFilesFor({
        cacheExclude: [ /main\.js$/ ]
      })).to.eventually.deep.equal(['core.js']);
    });

    it('returns an array of files keyed both include & exclude filters', function() {
      return expect(getListFilesFor({
        cacheInclude: [ /\.js$/ ],
        cacheExclude: [ /core\.js$/ ]
      })).to.eventually.deep.equal(['main.js']);
    });
  });

  describe('listEntries', function() {
    var listEntries;

    function getListFilesFor(options) {
      setupCachingWriter([sourcePath], options, function() {
        var writer = this;
        listEntries = this.listEntries().map(function(p) {
          return path.relative(writer.inputPaths[0], p.fullPath);
        });
      });

      return expectRebuild().then(function() {
        return listEntries;
      });
    }

    it('returns an array of files keyed', function() {
      return expect(getListFilesFor({})).to.eventually.deep.equal(['core.js', 'main.js']);
    });

    it('returns an array of files keyed including only those in the include filter', function() {
      return expect(getListFilesFor({
        cacheInclude: [ /core\.js$/ ]
      })).to.eventually.deep.equal(['core.js']);
    });

    it('returns an array of files keyed ignoring those in the exclude filter', function() {
      return expect(getListFilesFor({
        cacheExclude: [ /main\.js$/ ]
      })).to.eventually.deep.equal(['core.js']);
    });

    it('returns an array of files keyed both include & exclude filters', function() {
      return expect(getListFilesFor({
        cacheInclude: [ /\.js$/ ],
        cacheExclude: [ /core\.js$/ ]
      })).to.eventually.deep.equal(['main.js']);
    });
  });
});

var canUseInputFiles = require('../can-use-input-files');

describe('can-use-input-files', function(){
  it('is false for no input', function() {
    expect(canUseInputFiles()).to.eql(false);
  });

  it('is false for non array input', function() {
    expect(canUseInputFiles({})).to.eql(false);
    expect(canUseInputFiles({length: 1})).to.eql(false);
    expect(canUseInputFiles(true)).to.eql(false);
    expect(canUseInputFiles(false)).to.eql(false);
    expect(canUseInputFiles('')).to.eql(false);
    expect(canUseInputFiles('asdf')).to.eql(false);
  });

  it('is true for array input', function() {
    expect(canUseInputFiles([])).to.eql(true);
    expect(canUseInputFiles([1])).to.eql(true);
  });

  it('true for non glob entries', function() {
    expect(canUseInputFiles(['foo'])).to.eql(true);
    expect(canUseInputFiles(['foo', 'bar'])).to.eql(true);
    expect(canUseInputFiles(['foo/bar', 'bar/baz'])).to.eql(true);
    expect(canUseInputFiles(['foo/bar.js', 'bar/baz-apple'])).to.eql(true);
  });

  it('false for glob entries', function() {
    expect(canUseInputFiles(['f*oo'])).to.eql(false);
    expect(canUseInputFiles(['foo', 'bar*'])).to.eql(false);
    expect(canUseInputFiles(['foo/bar}', 'bar{baz'])).to.eql(false);
    expect(canUseInputFiles(['foo{bar.js', 'bar}baz{apple'])).to.eql(false);
  });
});
