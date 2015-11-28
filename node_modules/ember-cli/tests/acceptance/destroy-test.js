/*jshint quotmark: false*/

'use strict';

var Promise    = require('../../lib/ext/promise');
var expect     = require('chai').expect;
var assertFile = require('../helpers/assert-file');
var conf       = require('../helpers/conf');
var ember      = require('../helpers/ember');
var existsSync = require('exists-sync');
var fs         = require('fs-extra');
var outputFile = Promise.denodeify(fs.outputFile);
var path       = require('path');
var remove     = Promise.denodeify(fs.remove);
var root       = process.cwd();
var tmp        = require('tmp-sync');
var tmproot    = path.join(root, 'tmp');
var EOL        = require('os').EOL;

var BlueprintNpmTask = require('../helpers/disable-npm-on-blueprint');

describe('Acceptance: ember destroy', function() {
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
    this.timeout(10000);

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

  function initAddon() {
    return ember([
      'addon',
      'my-addon',
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

  function generate(args) {
    var generateArgs = ['generate'].concat(args);
    return ember(generateArgs);
  }

  function generateInAddon(args) {
    var generateArgs = ['generate'].concat(args);

    return initAddon().then(function() {
      return ember(generateArgs);
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

  function assertDestroyAfterGenerate(args, files) {
    return initApp()
      .then(function() {
        return generate(args);
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

  it('controller foo', function() {
    this.timeout(20000);
    var commandArgs = ['controller', 'foo'];
    var files       = [
      'app/controllers/foo.js',
      'tests/unit/controllers/foo-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('controller foo/bar', function() {
    this.timeout(20000);
    var commandArgs = ['controller', 'foo/bar'];
    var files       = [
      'app/controllers/foo/bar.js',
      'tests/unit/controllers/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('component x-foo', function() {
    this.timeout(20000);
    var commandArgs = ['component', 'x-foo'];
    var files       = [
      'app/components/x-foo.js',
      'app/templates/components/x-foo.hbs',
      'tests/integration/components/x-foo-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('helper foo-bar', function() {
    this.timeout(20000);
    var commandArgs = ['helper', 'foo-bar'];
    var files       = [
      'app/helpers/foo-bar.js',
      'tests/unit/helpers/foo-bar-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('helper foo/bar-baz', function() {
    this.timeout(20000);
    var commandArgs = ['helper', 'foo/bar-baz'];
    var files       = [
      'app/helpers/foo/bar-baz.js',
      'tests/unit/helpers/foo/bar-baz-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('model foo', function() {
    this.timeout(20000);
    var commandArgs = ['model', 'foo'];
    var files       = [
      'app/models/foo.js',
      'tests/unit/models/foo-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('model foo/bar', function() {
    this.timeout(20000);
    var commandArgs = ['model', 'foo/bar'];
    var files       = [
      'app/models/foo/bar.js',
      'tests/unit/models/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('route foo', function() {
    this.timeout(20000);
    var commandArgs = ['route', 'foo'];
    var files       = [
      'app/routes/foo.js',
      'app/templates/foo.hbs',
      'tests/unit/routes/foo-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files)
      .then(function() {
        assertFile('app/router.js', {
          doesNotContain: "this.route('foo');"
        });
      });
  });

    it('route foo --skip-router', function() {
    this.timeout(20000);
    var commandArgs = ['route', 'foo', '--skip-router'];
    var files       = [
      'app/routes/foo.js',
      'app/templates/foo.hbs',
      'tests/unit/routes/foo-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files)
      .then(function() {
        assertFile('app/router.js', {
          doesContain: "this.route('foo');"
        });
      });
  });

  it('route index', function() {
    this.timeout(20000);
    var commandArgs = ['route', 'index'];
    var files       = [
      'app/routes/index.js',
      'app/templates/index.hbs',
      'tests/unit/routes/index-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('route basic', function() {
    this.timeout(20000);
    var commandArgs = ['route', 'basic'];
    var files       = [
      'app/routes/basic.js',
      'app/templates/basic.hbs',
      'tests/unit/routes/basic-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('resource foo', function() {
    this.timeout(20000);
    var commandArgs = ['resource', 'foo'];
    var files       = [
      'app/models/foo.js',
      'tests/unit/models/foo-test.js',
      'app/routes/foo.js',
      'tests/unit/routes/foo-test.js',
      'app/templates/foo.hbs'
    ];

    return assertDestroyAfterGenerate(commandArgs, files)
      .then(function() {
        assertFile('app/router.js', {
          doesNotContain: "this.route('foo');"
        });
      });
  });

  it('resource foos', function() {
    this.timeout(20000);
    var commandArgs = ['resource', 'foos'];
    var files       = [
      'app/models/foo.js',
      'tests/unit/models/foo-test.js',
      'app/routes/foos.js',
      'tests/unit/routes/foos-test.js',
      'app/templates/foos.hbs'
    ];

    return assertDestroyAfterGenerate(commandArgs, files)
      .then(function() {
        assertFile('app/router.js', {
          doesNotContain: "this.route('foos');"
        });
      });
  });

  it('template foo', function() {
    this.timeout(20000);
    var commandArgs = ['template', 'foo'];
    var files       = ['app/templates/foo.hbs'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('template foo/bar', function() {
    this.timeout(20000);
    var commandArgs = ['template', 'foo/bar'];
    var files       = ['app/templates/foo/bar.hbs'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('view foo', function() {
    this.timeout(20000);
    var commandArgs = ['view', 'foo'];
    var files       = [
      'app/views/foo.js',
      'tests/unit/views/foo-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('view foo/bar', function() {
    this.timeout(20000);
    var commandArgs = ['view', 'foo/bar'];
    var files       = [
      'app/views/foo/bar.js',
      'tests/unit/views/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('initializer foo', function() {
    this.timeout(20000);
    var commandArgs = ['initializer', 'foo'];
    var files       = ['app/initializers/foo.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('initializer foo/bar', function() {
    this.timeout(20000);
    var commandArgs = ['initializer', 'foo/bar'];
    var files       = ['app/initializers/foo/bar.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('mixin foo', function() {
    this.timeout(20000);
    var commandArgs = ['mixin', 'foo'];
    var files       = [
      'app/mixins/foo.js',
      'tests/unit/mixins/foo-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('mixin foo/bar', function() {
    this.timeout(20000);
    var commandArgs = ['mixin', 'foo/bar'];
    var files       = [
      'app/mixins/foo/bar.js',
      'tests/unit/mixins/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('adapter foo', function() {
    this.timeout(20000);
    var commandArgs = ['adapter', 'foo'];
    var files       = ['app/adapters/foo.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('adapter foo/bar', function() {
    this.timeout(20000);
    var commandArgs = ['adapter', 'foo/bar'];
    var files       = ['app/adapters/foo/bar.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('serializer foo', function() {
    this.timeout(20000);
    var commandArgs = ['serializer', 'foo'];
    var files       = [
      'app/serializers/foo.js',
      'tests/unit/serializers/foo-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('serializer foo/bar', function() {
    this.timeout(20000);
    var commandArgs = ['serializer', 'foo/bar'];
    var files       = [
      'app/serializers/foo/bar.js',
      'tests/unit/serializers/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('transform foo', function() {
    this.timeout(20000);
    var commandArgs = ['transform', 'foo'];
    var files       = [
      'app/transforms/foo.js',
      'tests/unit/transforms/foo-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('transform foo/bar', function() {
    this.timeout(20000);
    var commandArgs = ['transform', 'foo/bar'];
    var files       = [
      'app/transforms/foo/bar.js',
      'tests/unit/transforms/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('util foo-bar', function() {
    this.timeout(20000);
    var commandArgs = ['util', 'foo-bar'];
    var files       = [
      'app/utils/foo-bar.js',
      'tests/unit/utils/foo-bar-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('util foo-bar/baz', function() {
    this.timeout(20000);
    var commandArgs = ['util', 'foo/bar-baz'];
    var files       = [
      'app/utils/foo/bar-baz.js',
      'tests/unit/utils/foo/bar-baz-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('service foo', function() {
    this.timeout(20000);
    var commandArgs = ['service', 'foo'];
    var files       = [
      'app/services/foo.js',
      'tests/unit/services/foo-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('service foo/bar', function() {
    this.timeout(20000);
    var commandArgs = ['service', 'foo/bar'];
    var files       = [
      'app/services/foo/bar.js',
      'tests/unit/services/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('blueprint foo', function() {
    this.timeout(20000);
    var commandArgs = ['blueprint', 'foo'];
    var files       = ['blueprints/foo/index.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('blueprint foo/bar', function() {
    this.timeout(20000);
    var commandArgs = ['blueprint', 'foo/bar'];
    var files       = ['blueprints/foo/bar/index.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('http-mock foo', function() {
    this.timeout(20000);
    var commandArgs = ['http-mock', 'foo'];
    var files       = ['server/mocks/foo.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('http-proxy foo', function() {
    this.timeout(20000);
    var commandArgs = ['http-proxy', 'foo', 'bar'];
    var files       = ['server/proxies/foo.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('in-addon component x-foo', function() {
    this.timeout(20000);
    var commandArgs = ['component', 'x-foo'];
    var files       = [
      'addon/components/x-foo.js',
      'addon/templates/components/x-foo.hbs',
      'app/components/x-foo.js',
      'tests/integration/components/x-foo-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
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

  it('acceptance-test foo', function() {
    this.timeout(20000);
    var commandArgs = ['acceptance-test', 'foo'];
    var files       = ['tests/acceptance/foo-test.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('deletes files generated using blueprints from the project directory', function() {
    this.timeout(20000);
    var commandArgs = ['foo', 'bar'];
    var files       = ['app/foos/bar.js'];
    return initApp()
      .then(function() {
        return outputFile(
          'blueprints/foo/files/app/foos/__name__.js',
          "import Ember from 'ember';" + EOL + EOL +
          'export default Ember.Object.extend({ foo: true });' + EOL
        );
      })
      .then(function() {
        return generate(commandArgs);
      })
      .then(function() {
        assertFilesExist(files);
      })
      .then(function() {
        return destroy(commandArgs);
      })
      .then(function() {
        assertFilesNotExist(files);
      });
  });

  it('correctly identifies the root of the project', function() {
    this.timeout(20000);
    var commandArgs = ['controller', 'foo'];
    var files       = ['app/controllers/foo.js'];
    return initApp()
      .then(function() {
        return outputFile(
          'blueprints/controller/files/app/controllers/__name__.js',
          "import Ember from 'ember';" + EOL + EOL +
          "export default Ember.Controller.extend({ custom: true });" + EOL
        );
      })
      .then(function() {
        return generate(commandArgs);
      })
      .then(function() {
        assertFilesExist(files);
      })
      .then(function() {
        process.chdir(path.join(tmpdir, 'app'));
      })
      .then(function() {
        return destroy(commandArgs);
      })
      .then(function() {
        process.chdir(tmpdir);
      })
      .then(function() {
        assertFilesNotExist(files);
      });
  });

  it('http-mock <name> does not remove server/', function() {
    this.timeout(20000);
    return initApp()
      .then(function() { return generate(['http-mock', 'foo']); })
      .then(function() { return generate(['http-mock', 'bar']); })
      .then(function() { return destroy(['http-mock', 'foo']); })
      .then(function() {
        assertFile('server/index.js');
        assertFile('server/.jshintrc');
      });
  });

});
