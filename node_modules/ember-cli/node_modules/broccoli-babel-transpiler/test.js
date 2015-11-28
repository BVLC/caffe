'use strict';

var fs     = require('fs');
var expect = require('chai').expect;
var broccoli = require('broccoli');
var path = require('path');
var Babel = require('./index');
var helpers = require('broccoli-test-helpers');
var stringify = require('json-stable-stringify');
var mkdirp = require('mkdirp').sync;
var rm = require('rimraf').sync;
var makeTestHelper = helpers.makeTestHelper;
var cleanupBuilders = helpers.cleanupBuilders;

var inputPath = path.join(__dirname, 'fixtures');
var expectations = path.join(__dirname, 'expectations');

var babel;

describe('options', function() {
  var options;

  before(function() {
    options = {
      foo: 1,
      bar: {
        baz: 1
      },
      filterExtensions: ['es6']
    };

    babel = new Babel('foo', options);
  });

  it('are cloned', function() {
    var transpilerOptions;

    babel.transform = function(string, options) {
      transpilerOptions = options;
      return { code: {} };
    }

    expect(transpilerOptions).to.eql(undefined);
    babel.processString('path', 'relativePath');

    expect(transpilerOptions.foo).to.eql(1);
    expect(transpilerOptions.bar.baz).to.eql(1);

    options.foo = 2;
    options.bar.baz = 2;

    expect(transpilerOptions.foo).to.eql(1);
    expect(transpilerOptions.bar.baz).to.eql(1);
  });

  it('correct fileName, sourceMapName, sourceFileName', function() {
    var transpilerOptions;

    babel.transform = function(string, options) {
      transpilerOptions = options;
      return { code: {} };
    }

    expect(transpilerOptions).to.eql(undefined);
    babel.processString('path', 'relativePath');

    expect(transpilerOptions.moduleId).to.eql(undefined);
    expect(transpilerOptions.filename).to.eql('relativePath');
    expect(transpilerOptions.sourceMapName).to.eql('relativePath');
    expect(transpilerOptions.sourceFileName).to.eql('relativePath');
  });

  it('includes moduleId if options.moduleId is true', function() {
    babel.options.moduleId = true;
    babel.options.filename = 'relativePath.es6';

    var transpilerOptions;

    babel.transform = function(string, options) {
      transpilerOptions = options;
      return { code: {} };
    }

    expect(transpilerOptions).to.eql(undefined);
    babel.processString('path', 'relativePath');

    expect(transpilerOptions.moduleId).to.eql('relativePath');
  });

  it('does not propagate validExtensions', function () {
    var transpilerOptions;

    babel.transform = function(string, options) {
      transpilerOptions = options;
      return { code: {} };
    };

    expect(transpilerOptions).to.eql(undefined);
    babel.processString('path', 'relativePath');

    expect(transpilerOptions.filterExtensions).to.eql(undefined);
  });
});

describe('transpile ES6 to ES5', function() {

  before(function() {
    babel = makeTestHelper({
      subject: function() {
        return new Babel(arguments[0], arguments[1]);
      },
      fixturePath: inputPath
    });
  });


  afterEach(function () {
    return cleanupBuilders();
  });

  it('basic', function () {
    return babel('files', {
      inputSourceMap:false,
      sourceMap: false
    }).then(function(results) {
      var outputPath = results.directory;

      var output = fs.readFileSync(path.join(outputPath, 'fixtures.js')).toString();
      var input = fs.readFileSync(path.join(expectations,  'expected.js')).toString();

      expect(output).to.eql(input);
    });
  });

  it('inline source maps', function () {
    return babel('files', {
      sourceMap: 'inline'
    }).then(function(results) {
      var outputPath = results.directory;

      var output = fs.readFileSync(path.join(outputPath, 'fixtures.js')).toString();
      var input = fs.readFileSync(path.join(expectations,  'expected-inline-source-maps.js')).toString();

      expect(output).to.eql(input);
    });
  });
});

describe('filters files to transform', function() {

  before(function() {
    babel = makeTestHelper({
      subject: function() {
        return new Babel(arguments[0], arguments[1]);
      },
      fixturePath: inputPath
    });
  });

  afterEach(function () {
    return cleanupBuilders();
  });

  it('default', function () {
    return babel('files', {
      inputSourceMap:false,
      sourceMap: false
    }).then(function(results) {
      var outputPath = results.directory;

      var output = fs.readFileSync(path.join(outputPath, 'fixtures.js')).toString();
      var input = fs.readFileSync(path.join(expectations,  'expected.js')).toString();

      expect(output).to.eql(input);
      // Verify that .es6 file was not transformed
      expect(fs.existsSync(path.join(outputPath, 'fixtures-es6.es6'))).to.be.ok;

    });
  });

  it('uses specified filter', function () {
    return babel('files', {
      filterExtensions: ['es6'],
      inputSourceMap: false,
      sourceMap: false
    }).then(function(results) {
      var outputPath = results.directory;

      var output = fs.readFileSync(path.join(outputPath, 'fixtures-es6.js')).toString();
      var input = fs.readFileSync(path.join(expectations,  'expected.js')).toString();

      expect(output).to.eql(input);
      // Verify that .es6 file was not transformed
      expect(fs.existsSync(path.join(outputPath, 'fixtures-es6.es6'))).to.not.be.ok;

    });
  });

  it('uses multiple specified filters', function() {
    return babel('files', {
      filterExtensions: ['js', 'es6'],
      inputSourceMap: false,
      sourceMap: false
    }).then(function(results) {
      var outputPath = results.directory;

      var es6ExtOutput = fs.readFileSync(path.join(outputPath, 'fixtures-es6.js')).toString();
      var jsExtOutput = fs.readFileSync(path.join(outputPath, 'fixtures.js')).toString();
      var input = fs.readFileSync(path.join(expectations, 'expected.js')).toString();

      expect(es6ExtOutput).to.eql(input);
      expect(jsExtOutput).to.eql(input);
      // Verify that .es6 file was not transformed
      expect(fs.existsSync(path.join(outputPath, 'fixtures-es6.es6'))).to.not.be.ok;
    });
  });

  it('named module', function() {
    return babel('files', {
      inputSourceMap: false,
      sourceMap: false,
      moduleId: "foo",
      modules: 'amdStrict'
    }).then(function(results) {
      var outputPath = results.directory;

      var output = fs.readFileSync(path.join(outputPath, 'named-module-fixture.js')).toString();
      var input = fs.readFileSync(path.join(expectations,  'named-module.js')).toString();

      expect(output).to.eql(input);
    });
  });


  it('moduleId === true', function() {
    return babel('files', {
      inputSourceMap: false,
      sourceMap: false,
      moduleId: true,
      modules: 'amdStrict'
    }).then(function(results) {
      var outputPath = results.directory;

      var output = fs.readFileSync(path.join(outputPath, 'true-module-fixture.js')).toString();
      var input = fs.readFileSync(path.join(expectations,  'true-module.js')).toString();

      expect(output).to.eql(input);
    });
  });
});

// describe('module metadata', function() {
//   before(function() {
//     babel = makeTestHelper({
//       subject: function() {
//         return new Babel(arguments[0], arguments[1]);
//       },
//       fixturePath: inputPath
//     });
//   });

//   afterEach(function () {
//     return cleanupBuilders();
//   });

//   it('exports module metadata', function() {
//     return babel('files', {
//       exportModuleMetadata: true,
//       moduleId: true,
//       modules: 'amdStrict',
//       sourceMap: false,
//       inputSourceMap: false
//     }).then(function(results) {
//       var outputPath = results.directory;
//       var output = fs.readFileSync(path.join(outputPath, 'dep-graph.json'), 'utf8');
//       var expectation = fs.readFileSync(path.join(expectations, 'dep-graph.json'), 'utf8');
//       expect(output).to.eql(expectation);
//     });
//   });

//   it('handles adding and removing files', function() {
//     return babel('files', {
//       exportModuleMetadata: true,
//       moduleId: true,
//       modules: 'amdStrict',
//       sourceMap: false,
//       inputSourceMap: false
//     }).then(function(results) {
//       // Normal build
//       var outputPath = results.directory;
//       var output = fs.readFileSync(path.join(outputPath , 'dep-graph.json'), 'utf8');
//       var expectation = fs.readFileSync(path.join(expectations, 'dep-graph.json'), 'utf8');
//       expect(output).to.eql(expectation);

//       // Move away files/fixtures.js
//       fs.renameSync(path.join(inputPath, 'files', 'fixtures.js'), path.join(inputPath, 'fixtures.js'));
//       return results.builder();
//     }).then(function(results) {
//       // Add back file/fixtures.js
//       fs.renameSync(path.join(inputPath, 'fixtures.js'), path.join(inputPath, 'files', 'fixtures.js'));

//       // Build without files/fixtures.js
//       var outputPath = results.directory;
//       var output = fs.readFileSync(path.join(outputPath, 'dep-graph.json'), 'utf8');
//       var expectation = fs.readFileSync(path.join(expectations, 'pruned-dep-graph.json'), 'utf8');
//       expect(output).to.eql(expectation);

//       return results.builder();
//     }).then(function(results) {
//       // Back to the first build
//       var outputPath = results.directory;
//       var output = fs.readFileSync(path.join(outputPath, 'dep-graph.json'), 'utf8');
//       var expectation = fs.readFileSync(path.join(expectations, 'dep-graph.json'), 'utf8');
//       expect(output).to.eql(expectation);
//     });
//   });

//   describe('_generateDepGraph', function() {
//     var tmp = path.join(process.cwd(), 'test-temp');
//     beforeEach(function() {
//       mkdirp(tmp);
//       babel = new Babel('foo');
//       babel.outputPath = tmp;
//     });

//     afterEach(function() {
//       rm(tmp);
//       babel.outputPath = null;
//     });

//     it('should generate a graph', function() {
//       babel._cache.keys = function() {
//         return ['foo.js', 'bar.js'];
//       };

//       babel.moduleMetadata = {
//         foo: {},
//         bar: {}
//       };

//       babel._generateDepGraph();

//       expect(fs.readFileSync(path.join(babel.outputPath, 'dep-graph.json'), 'utf8')).to.eql(stringify({
//         bar: {},
//         foo: {}
//       }, { space: 2 }));
//     });

//     it('should evict imports from the graph that are no longer in the tree', function() {
//       babel._cache.keys = function() {
//         return ['foo.js'];
//       };

//       babel.moduleMetadata = {
//         foo: {}
//       };

//       babel._generateDepGraph();

//       expect(fs.readFileSync(path.join(babel.outputPath, 'dep-graph.json'), 'utf8')).to.eql(stringify({
//         foo: {}
//       }, { space: 2 }));
//     });
//   });

// });

describe('consume broccoli-babel-transpiler options', function() {
  it('enabled', function() {
    var options = {
      exportModuleMetadata: true,
      browserPolyfill: true
    };

    babel = new Babel('foo', options);
    var code = babel.processString('path', 'relativePath');
    expect(code).to.be.ok;
  });

  it('explicitly disabled', function() {
    var options = {
      exportModuleMetadata: false,
      browserPolyfill: false
    };

    babel = new Babel('foo', options);
    var code = babel.processString('path', 'relativePath');
    expect(code).to.be.ok;
  });
});
