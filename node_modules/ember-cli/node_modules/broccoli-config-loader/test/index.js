var assert = require('assert'),
    broccoli = require('broccoli'),
    ConfigLoader = require('..'),
    Project = require('./fixtures/project'),
    join = require('path').join,
    fs = require('fs');

var createBuilder = function(options) {
  var configDir = join(options.project.root, 'config'),
      loader = new ConfigLoader(configDir, options);
  return new broccoli.Builder(loader);
};

var expectResult = function(name, expected) {
  return function(result) {
    var path = join(result.directory, 'environments', name),
        contents = fs.readFileSync(path, { encoding: 'utf8' }),
        actual = JSON.parse(contents);

    assert.deepEqual(actual, expected);
  };
};

afterEach(function() {
  if (this.builder) {
    this.builder.cleanup();
  }
});

describe('config-loader', function() {
  it('writes development.json', function(done) {
    this.project = new Project({ foo: 'bar', baz: 'qux' });
    this.builder = createBuilder({
      env: 'development',
      tests: true,
      project: this.project
    });

    this.builder.build().then(
      expectResult('development.json', { foo: 'bar', baz: 'qux' })
    ).then(done).catch(console.log);
  });

  it('tests: true writes both development.json & test.json', function(done) {
    this.project = new Project({ apple: true, orange: 4 });
    this.builder = createBuilder({
      env: 'development',
      tests: true,
      project: this.project
    });

    this.builder.build().then(function(result) {
      var dev = join(result.directory, 'environments', 'development.json'),
          test = join(result.directory, 'environments', 'test.json');

      assert.ok(fs.statSync(dev), 'development.json exists');
      assert.ok(fs.statSync(test), 'test.json exists');
      done();
    }).catch(console.log);
  });

  it('clearConfigGeneratorCache expires the cache', function(done) {
    var project = this.project = new Project({ name: 'Max', age: 12 }),
        builder = this.builder = createBuilder({
          env: 'development',
          tests: true,
          project: this.project
        });

    builder.build().then(
      expectResult('development.json', { name: 'Max', age: 12 })
    ).then(function() {
      project.writeConfig({ name: 'Maxine', age: 12 });
      return builder.build();
    }).then(
      expectResult('development.json', { name: 'Maxine', age: 12 })
    ).then(done).catch(console.log);
  });

  it('does not generate test environment files if testing is disabled', function(done) {
    this.project = new Project({ apple: true, orange: 4 });
    this.builder = createBuilder({
      env: 'development',
      tests: false,
      project: this.project
    });

    this.builder.build().then(function(result) {
      var test = join(result.directory, 'environments', 'test.json');
      assert.throws(function() {
        fs.statSync(test);
      }, 'test.json should not exist');
      done();
    }).catch(console.log);
  });
});
