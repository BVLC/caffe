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

describe('Acceptance: ember destroy in-repo-addon', function() {
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

  function initApp() {
    return ember([
      'init',
      '--name=my-app',
      '--skip-npm',
      '--skip-bower'
    ]);
  }

  function initInRepoAddon() {
    return initApp().then(function() {
      return ember([
        'generate',
        'in-repo-addon',
        'my-addon'
      ]);
    });
  }

  function generateInRepoAddon(args) {
    var generateArgs = ['generate'].concat(args);

    return initInRepoAddon().then(function() {
      return ember(generateArgs);
    });
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

  function assertDestroyAfterGenerateInRepoAddon(args, files) {
    return generateInRepoAddon(args)
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

  it('in-repo-addon controller foo', function() {
    var commandArgs = ['controller', 'foo', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/controllers/foo.js',
      'lib/my-addon/app/controllers/foo.js',
      'tests/unit/controllers/foo-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon controller foo/bar', function() {
    var commandArgs = ['controller', 'foo/bar', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/controllers/foo/bar.js',
      'lib/my-addon/app/controllers/foo/bar.js',
      'tests/unit/controllers/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon component x-foo', function() {
    var commandArgs = ['component', 'x-foo', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/components/x-foo.js',
      'lib/my-addon/addon/templates/components/x-foo.hbs',
      'lib/my-addon/app/components/x-foo.js',
      'tests/integration/components/x-foo-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon component nested/x-foo', function() {
    var commandArgs = ['component', 'nested/x-foo', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/components/nested/x-foo.js',
      'lib/my-addon/addon/templates/components/nested/x-foo.hbs',
      'lib/my-addon/app/components/nested/x-foo.js',
      'tests/integration/components/nested/x-foo-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon helper foo-bar', function() {
    var commandArgs = ['helper', 'foo-bar', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/helpers/foo-bar.js',
      'lib/my-addon/app/helpers/foo-bar.js',
      'tests/unit/helpers/foo-bar-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon helper foo/bar-baz', function() {
    var commandArgs = ['helper', 'foo/bar-baz', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/helpers/foo/bar-baz.js',
      'lib/my-addon/app/helpers/foo/bar-baz.js',
      'tests/unit/helpers/foo/bar-baz-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon model foo', function() {
    var commandArgs = ['model', 'foo', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/models/foo.js',
      'lib/my-addon/app/models/foo.js',
      'tests/unit/models/foo-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon model foo/bar', function() {
    var commandArgs = ['model', 'foo/bar', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/models/foo/bar.js',
      'lib/my-addon/app/models/foo/bar.js',
      'tests/unit/models/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });
/*
  it('in-repo-addon route foo', function() {
    var commandArgs = ['route', 'foo'];
    var files       = [
      'lib/my-addon/app/routes/foo.js',
      'lib/my-addon/app/templates/foo.hbs',
      'tests/unit/routes/foo-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files)
      .then(function() {
        assertFile('lib/my-addon/app/router.js', {
          doesNotContain: "this.route('foo');"
        });
      });
  });

  it('in-repo-addon route index', function() {
    var commandArgs = ['route', 'index'];
    var files       = [
      'lib/my-addon/app/routes/index.js',
      'lib/my-addon/app/templates/index.hbs',
      'tests/unit/routes/index-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon route basic', function() {
    var commandArgs = ['route', 'basic'];
    var files       = [
      'lib/my-addon/app/routes/basic.js',
      'lib/my-addon/app/templates/basic.hbs',
      'tests/unit/routes/basic-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon resource foo', function() {
    var commandArgs = ['resource', 'foo'];
    var files       = [
      'lib/my-addon/app/models/foo.js',
      'tests/unit/models/foo-test.js',
      'lib/my-addon/app/routes/foo.js',
      'tests/unit/routes/foo-test.js',
      'lib/my-addon/app/templates/foo.hbs'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files)
      .then(function() {
        assertFile('lib/my-addon/app/router.js', {
          doesNotContain: "this.route('foo');"
        });
      });
  });

  it('in-repo-addon resource foos', function() {
    var commandArgs = ['resource', 'foos'];
    var files       = [
      'lib/my-addon/app/models/foo.js',
      'tests/unit/models/foo-test.js',
      'lib/my-addon/app/routes/foos.js',
      'tests/unit/routes/foos-test.js',
      'lib/my-addon/app/templates/foos.hbs'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files)
      .then(function() {
        assertFile('lib/my-addon/app/router.js', {
          doesNotContain: "this.route('foos');"
        });
      });
  });
*/
  it('in-repo-addon template foo', function() {
    var commandArgs = ['template', 'foo', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/templates/foo.hbs',
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon template foo/bar', function() {
    var commandArgs = ['template', 'foo/bar', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/templates/foo/bar.hbs',
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon view foo', function() {
    var commandArgs = ['view', 'foo', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/views/foo.js',
      'lib/my-addon/app/views/foo.js',
      'tests/unit/views/foo-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon view foo/bar', function() {
    var commandArgs = ['view', 'foo/bar', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/views/foo/bar.js',
      'lib/my-addon/app/views/foo/bar.js',
      'tests/unit/views/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon initializer foo', function() {
    var commandArgs = ['initializer', 'foo', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/initializers/foo.js',
      'lib/my-addon/app/initializers/foo.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon initializer foo/bar', function() {
    var commandArgs = ['initializer', 'foo/bar', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/initializers/foo/bar.js',
      'lib/my-addon/app/initializers/foo/bar.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon mixin foo', function() {
    var commandArgs = ['mixin', 'foo', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/mixins/foo.js',
      'tests/unit/mixins/foo-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon mixin foo/bar', function() {
    var commandArgs = ['mixin', 'foo/bar', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/mixins/foo/bar.js',
      'tests/unit/mixins/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon adapter foo', function() {
    var commandArgs = ['adapter', 'foo', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/adapters/foo.js',
      'lib/my-addon/app/adapters/foo.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon adapter foo/bar', function() {
    var commandArgs = ['adapter', 'foo/bar', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/adapters/foo/bar.js',
      'lib/my-addon/app/adapters/foo/bar.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon serializer foo', function() {
    var commandArgs = ['serializer', 'foo', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/serializers/foo.js',
      'lib/my-addon/app/serializers/foo.js',
      'tests/unit/serializers/foo-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon serializer foo/bar', function() {
    var commandArgs = ['serializer', 'foo/bar', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/serializers/foo/bar.js',
      'lib/my-addon/app/serializers/foo/bar.js',
      'tests/unit/serializers/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon transform foo', function() {
    var commandArgs = ['transform', 'foo', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/transforms/foo.js',
      'lib/my-addon/app/transforms/foo.js',
      'tests/unit/transforms/foo-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon transform foo/bar', function() {
    var commandArgs = ['transform', 'foo/bar', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/transforms/foo/bar.js',
      'lib/my-addon/app/transforms/foo/bar.js',
      'tests/unit/transforms/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon util foo-bar', function() {
    var commandArgs = ['util', 'foo-bar', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/utils/foo-bar.js',
      'lib/my-addon/app/utils/foo-bar.js',
      'tests/unit/utils/foo-bar-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon util foo-bar/baz', function() {
    var commandArgs = ['util', 'foo/bar-baz', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/utils/foo/bar-baz.js',
      'lib/my-addon/app/utils/foo/bar-baz.js',
      'tests/unit/utils/foo/bar-baz-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon service foo', function() {
    var commandArgs = ['service', 'foo', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/services/foo.js',
      'lib/my-addon/app/services/foo.js',
      'tests/unit/services/foo-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon service foo/bar', function() {
    var commandArgs = ['service', 'foo/bar', '--in-repo-addon=my-addon'];
    var files       = [
      'lib/my-addon/addon/services/foo/bar.js',
      'lib/my-addon/app/services/foo/bar.js',
      'tests/unit/services/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

});
