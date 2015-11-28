var assert = require('assert'),
    broccoli = require('broccoli'),
    ConfigReplace = require('..'),
    join = require('path').join,
    fs = require('fs'),
    tmp = require('tmp-sync');

afterEach(function() {
  if (this.builder) {
    this.builder.cleanup();
  }
});

function writeExample(options) {
  var root = tmp.in(join(process.cwd(), 'tmp'));
  fs.writeFileSync(join(root, 'config.json'), options.config);
  fs.writeFileSync(join(root, 'index.html'), options.index);
  return root;
};

function makeConfigReplace(root, patterns) {
  return new ConfigReplace(
    // In these tests, config.json and index.html live in the same
    // directory, so we just pass the root twice.
    root,
    root, {
      files: ['index.html'],
      configPath: 'config.json',
      patterns: patterns
    }
  );
};

function makeBuilder(root, patterns) {
  var configReplace = makeConfigReplace(root, patterns);
  return new broccoli.Builder(configReplace);
};

var expectEquals = function(expected) {
  return function(results) {
    var resultsPath = join(results.directory, 'index.html'),
        contents = fs.readFileSync(resultsPath, { encoding: 'utf8' });

    assert.equal(contents.trim(), expected);
  };
};

describe('config-replace', function() {
  it('replaces with text from config.json', function(done) {
    var root = writeExample({
      config: '{"color":"red"}',
      index: '{{color}}'
    });

    makeBuilder(root, [{
      match: /\{\{color\}\}/g,
      replacement: function(config) { return config.color; }
    }]).build().then(
      expectEquals('red')
    ).then(done).catch(console.log);
  });

  it('replaces with string passed in via options', function(done) {
    var root = writeExample({
      config: '{}',
      index: '{{name}}'
    });

    makeBuilder(root, [{
      match: /\{\{name\}\}/g,
      replacement: 'hari'
    }]).build().then(
      expectEquals('hari')
    ).then(done).catch(console.log);
  });

  it('rebuilds if the config file changes', function(done) {
    var root = writeExample({
      config: '{"pokemon":"diglet"}',
      index: '{{pokemon}}'
    });

    var builder = makeBuilder(root, [{
      match: /\{\{pokemon\}\}/g,
      replacement: function(config) { return config.pokemon; }
    }]);

    builder.build().then(
      expectEquals('diglet')
    ).then(function() {
      fs.writeFileSync(join(root, 'config.json'), '{"pokemon":"jigglypuff"}');
      return builder.build();
    }).then(
      expectEquals('jigglypuff')
    ).then(done).catch(console.log);
  });

  it('caches the result', function(done) {
    var root, configReplace, builder, key, entry;

    root = writeExample({
      config: '{"city":"nyc"}',
      index: '{{city}}'
    });

    configReplace = makeConfigReplace(root, [{
      match: /\{\{city\}\}/g,
      replacement: function(config) { return config.city; }
    }]);

    builder = new broccoli.Builder(configReplace);
    builder.build().then(
      expectEquals('nyc')
    ).then(function() {
      key = Object.keys(configReplace._cache)[0];
      entry = configReplace._cache[key];
      return builder.build();
    }).then(function() {
      assert.equal(entry, configReplace._cache[key]);
    }).then(done).catch(console.log);
  });
});
