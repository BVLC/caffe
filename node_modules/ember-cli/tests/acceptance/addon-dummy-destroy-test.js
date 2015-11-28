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

describe('Acceptance: ember destroy in-addon-dummy', function() {
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

  function assertDestroyAfterGenerateInAddonDummy(args, files) {
    args = args.concat('--dummy');

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

  it('in-addon-dummy controller foo', function() {
    var commandArgs = ['controller', 'foo'];
    var files       = [
      'tests/dummy/app/controllers/foo.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy controller foo/bar', function() {
    var commandArgs = ['controller', 'foo/bar'];
    var files       = [
      'tests/dummy/app/controllers/foo/bar.js',
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy route foo', function() {
    var commandArgs = ['route', 'foo'];
    var files       = [
      'tests/dummy/app/routes/foo.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files)
      .then(function() {
        assertFile('tests/dummy/app/router.js', {
          doesNotContain: "this.route('foo');"
        });
      });
  });

  it('in-addon-dummy route foo/bar', function() {
    var commandArgs = ['route', 'foo/bar'];
    var files       = [
      'tests/dummy/app/routes/foo/bar.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files)
      .then(function() {
        assertFile('tests/dummy/app/router.js', {
          doesNotContain: "this.route('bar');"
        });
      });
  });

  it('in-addon-dummy component x-foo', function() {
    var commandArgs = ['component', 'x-foo'];
    var files       = [
      'tests/dummy/app/templates/components/x-foo.hbs',
      'tests/dummy/app/components/x-foo.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy helper foo-bar', function() {
    var commandArgs = ['helper', 'foo-bar'];
    var files       = [
      'tests/dummy/app/helpers/foo-bar.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy helper foo/bar-baz', function() {
    var commandArgs = ['helper', 'foo/bar-baz'];
    var files       = [
      'tests/dummy/app/helpers/foo/bar-baz.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy model foo', function() {
    var commandArgs = ['model', 'foo'];
    var files       = [
      'tests/dummy/app/models/foo.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy model foo/bar', function() {
    var commandArgs = ['model', 'foo/bar'];
    var files       = [
      'tests/dummy/app/models/foo/bar.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy template foo', function() {
    var commandArgs = ['template', 'foo'];
    var files       = [
      'tests/dummy/app/templates/foo.hbs',
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy template foo/bar', function() {
    var commandArgs = ['template', 'foo/bar'];
    var files       = [
      'tests/dummy/app/templates/foo/bar.hbs',
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy view foo', function() {
    var commandArgs = ['view', 'foo'];
    var files       = [
      'tests/dummy/app/views/foo.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy view foo/bar', function() {
    var commandArgs = ['view', 'foo/bar'];
    var files       = [
      'tests/dummy/app/views/foo/bar.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy initializer foo', function() {
    var commandArgs = ['initializer', 'foo'];
    var files       = [
      'tests/dummy/app/initializers/foo.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy initializer foo/bar', function() {
    var commandArgs = ['initializer', 'foo/bar'];
    var files       = [
      'tests/dummy/app/initializers/foo/bar.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy mixin foo', function() {
    var commandArgs = ['mixin', 'foo'];
    var files       = [
      'tests/dummy/app/mixins/foo.js',
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy mixin foo/bar', function() {
    var commandArgs = ['mixin', 'foo/bar'];
    var files       = [
      'tests/dummy/app/mixins/foo/bar.js',
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy adapter foo', function() {
    var commandArgs = ['adapter', 'foo'];
    var files       = [
      'tests/dummy/app/adapters/foo.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy adapter foo/bar', function() {
    var commandArgs = ['adapter', 'foo/bar'];
    var files       = [
      'tests/dummy/app/adapters/foo/bar.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy serializer foo', function() {
    var commandArgs = ['serializer', 'foo'];
    var files       = [
      'tests/dummy/app/serializers/foo.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy serializer foo/bar', function() {
    var commandArgs = ['serializer', 'foo/bar'];
    var files       = [
      'tests/dummy/app/serializers/foo/bar.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy transform foo', function() {
    var commandArgs = ['transform', 'foo'];
    var files       = [
      'tests/dummy/app/transforms/foo.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy transform foo/bar', function() {
    var commandArgs = ['transform', 'foo/bar'];
    var files       = [
      'tests/dummy/app/transforms/foo/bar.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy util foo-bar', function() {
    var commandArgs = ['util', 'foo-bar'];
    var files       = [
      'tests/dummy/app/utils/foo-bar.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy util foo-bar/baz', function() {
    var commandArgs = ['util', 'foo/bar-baz'];
    var files       = [
      'tests/dummy/app/utils/foo/bar-baz.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy service foo', function() {
    var commandArgs = ['service', 'foo'];
    var files       = [
      'tests/dummy/app/services/foo.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });

  it('in-addon-dummy service foo/bar', function() {
    var commandArgs = ['service', 'foo/bar'];
    var files       = [
      'tests/dummy/app/services/foo/bar.js'
    ];

    return assertDestroyAfterGenerateInAddonDummy(commandArgs, files);
  });
});
