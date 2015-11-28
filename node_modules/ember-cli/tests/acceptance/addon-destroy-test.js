/*jshint quotmark: false*/

'use strict';

var Promise    = require('../../lib/ext/promise');
var expect     = require('chai').expect;
var assertFile = require('../helpers/assert-file');
var conf       = require('../helpers/conf');
var ember      = require('../helpers/ember');
var existsSync = require('exists-sync');
var fs         = require('fs-extra');
var path       = require('path');
var remove     = Promise.denodeify(fs.remove);
var root       = process.cwd();
var tmp        = require('tmp-sync');
var tmproot    = path.join(root, 'tmp');

var BlueprintNpmTask = require('../helpers/disable-npm-on-blueprint');

describe('Acceptance: ember destroy in-addon', function() {
  this.timeout(20000);

  var tmpdir;

  before(function() {
    BlueprintNpmTask.disableNPM();
    conf.setup();
  });

  after(function() {
    BlueprintNpmTask.restoreNPM();
    conf.restore();
  });

  beforeEach(function() {
    tmpdir = tmp.in(tmproot);
    process.chdir(tmpdir);
  });

  afterEach(function() {
    process.chdir(root);
    return remove(tmproot);
  });

  function initAddon() {
    return ember([
      'addon',
      'my-addon',
      '--skip-npm',
      '--skip-bower'
    ]);
  }

  function generateInAddon(args) {
    var generateArgs = ['generate'].concat(args);

    return ember(generateArgs);
  }

  function destroy(args) {
    var destroyArgs = ['destroy'].concat(args);
    return ember(destroyArgs);
  }

  function assertFileNotExists(file) {
    var filePath = path.join(process.cwd(), file);
    expect(!existsSync(filePath), 'expected ' + file + ' not to exist');
  }

  function assertFilesExist(files) {
    files.forEach(assertFile);
  }

  function assertFilesNotExist(files) {
    files.forEach(assertFileNotExists);
  }

  function assertDestroyAfterGenerateInAddon(args, files) {
    return initAddon()
      .then(function() {
        return generateInAddon(args);
      })
      .then(function() {
        assertFilesExist(files);
      })
      .then(function() {
        return destroy(args);
      })
      .then(function(result) {
        expect(result, 'destroy command did not exit with errorCode').to.be.an('object');
        assertFilesNotExist(files);
      });
  }

  it('in-addon controller foo', function() {
    var commandArgs = ['controller', 'foo'];
    var files       = [
      'addon/controllers/foo.js',
      'app/controllers/foo.js',
      'tests/unit/controllers/foo-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon controller foo/bar', function() {
    var commandArgs = ['controller', 'foo/bar'];
    var files       = [
      'addon/controllers/foo/bar.js',
      'app/controllers/foo/bar.js',
      'tests/unit/controllers/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon component x-foo', function() {
    var commandArgs = ['component', 'x-foo'];
    var files       = [
      'addon/components/x-foo.js',
      'addon/templates/components/x-foo.hbs',
      'app/components/x-foo.js',
      'tests/integration/components/x-foo-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon helper foo-bar', function() {
    var commandArgs = ['helper', 'foo-bar'];
    var files       = [
      'addon/helpers/foo-bar.js',
      'app/helpers/foo-bar.js',
      'tests/unit/helpers/foo-bar-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon helper foo/bar-baz', function() {
    var commandArgs = ['helper', 'foo/bar-baz'];
    var files       = [
      'addon/helpers/foo/bar-baz.js',
      'app/helpers/foo/bar-baz.js',
      'tests/unit/helpers/foo/bar-baz-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon model foo', function() {
    var commandArgs = ['model', 'foo'];
    var files       = [
      'addon/models/foo.js',
      'app/models/foo.js',
      'tests/unit/models/foo-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon model foo/bar', function() {
    var commandArgs = ['model', 'foo/bar'];
    var files       = [
      'addon/models/foo/bar.js',
      'app/models/foo/bar.js',
      'tests/unit/models/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon template foo', function() {
    var commandArgs = ['template', 'foo'];
    var files       = [
      'addon/templates/foo.hbs',
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon template foo/bar', function() {
    var commandArgs = ['template', 'foo/bar'];
    var files       = [
      'addon/templates/foo/bar.hbs',
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon view foo', function() {
    var commandArgs = ['view', 'foo'];
    var files       = [
      'addon/views/foo.js',
      'app/views/foo.js',
      'tests/unit/views/foo-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon view foo/bar', function() {
    var commandArgs = ['view', 'foo/bar'];
    var files       = [
      'addon/views/foo/bar.js',
      'app/views/foo/bar.js',
      'tests/unit/views/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon initializer foo', function() {
    var commandArgs = ['initializer', 'foo'];
    var files       = [
      'addon/initializers/foo.js',
      'app/initializers/foo.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon initializer foo/bar', function() {
    var commandArgs = ['initializer', 'foo/bar'];
    var files       = [
      'addon/initializers/foo/bar.js',
      'app/initializers/foo/bar.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon mixin foo', function() {
    var commandArgs = ['mixin', 'foo'];
    var files       = [
      'addon/mixins/foo.js',
      'tests/unit/mixins/foo-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon mixin foo/bar', function() {
    var commandArgs = ['mixin', 'foo/bar'];
    var files       = [
      'addon/mixins/foo/bar.js',
      'tests/unit/mixins/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon adapter foo', function() {
    var commandArgs = ['adapter', 'foo'];
    var files       = [
      'addon/adapters/foo.js',
      'app/adapters/foo.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon adapter foo/bar', function() {
    var commandArgs = ['adapter', 'foo/bar'];
    var files       = [
      'addon/adapters/foo/bar.js',
      'app/adapters/foo/bar.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon serializer foo', function() {
    var commandArgs = ['serializer', 'foo'];
    var files       = [
      'addon/serializers/foo.js',
      'app/serializers/foo.js',
      'tests/unit/serializers/foo-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon serializer foo/bar', function() {
    var commandArgs = ['serializer', 'foo/bar'];
    var files       = [
      'addon/serializers/foo/bar.js',
      'app/serializers/foo/bar.js',
      'tests/unit/serializers/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon transform foo', function() {
    var commandArgs = ['transform', 'foo'];
    var files       = [
      'addon/transforms/foo.js',
      'app/transforms/foo.js',
      'tests/unit/transforms/foo-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon transform foo/bar', function() {
    var commandArgs = ['transform', 'foo/bar'];
    var files       = [
      'addon/transforms/foo/bar.js',
      'app/transforms/foo/bar.js',
      'tests/unit/transforms/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon util foo-bar', function() {
    var commandArgs = ['util', 'foo-bar'];
    var files       = [
      'addon/utils/foo-bar.js',
      'app/utils/foo-bar.js',
      'tests/unit/utils/foo-bar-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon util foo-bar/baz', function() {
    var commandArgs = ['util', 'foo/bar-baz'];
    var files       = [
      'addon/utils/foo/bar-baz.js',
      'app/utils/foo/bar-baz.js',
      'tests/unit/utils/foo/bar-baz-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon service foo', function() {
    var commandArgs = ['service', 'foo'];
    var files       = [
      'addon/services/foo.js',
      'app/services/foo.js',
      'tests/unit/services/foo-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-addon service foo/bar', function() {
    var commandArgs = ['service', 'foo/bar'];
    var files       = [
      'addon/services/foo/bar.js',
      'app/services/foo/bar.js',
      'tests/unit/services/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

});
