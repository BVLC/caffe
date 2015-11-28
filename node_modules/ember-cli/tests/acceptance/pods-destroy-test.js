/*jshint quotmark: false*/

'use strict';

var Promise     = require('../../lib/ext/promise');
var expect      = require('chai').expect;
var assertFile  = require('../helpers/assert-file');
var conf        = require('../helpers/conf');
var ember       = require('../helpers/ember');
var existsSync  = require('exists-sync');
var fs          = require('fs-extra');
var replaceFile = require('../helpers/file-utils').replaceFile;
var outputFile  = Promise.denodeify(fs.outputFile);
var path        = require('path');
var remove      = Promise.denodeify(fs.remove);
var root        = process.cwd();
var tmp         = require('tmp-sync');
var tmproot     = path.join(root, 'tmp');
var EOL         = require('os').EOL;

var BlueprintNpmTask = require('../helpers/disable-npm-on-blueprint');

describe('Acceptance: ember destroy pod', function() {
  var tmpdir;

  this.timeout(20000);

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
        replaceFile('config/environment.js', "var ENV = {", "var ENV = {" + EOL + "podModulePrefix: 'app/pods', " + EOL);
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

  function assertDestroyAfterGenerateWithUsePods(args, files) {
    return initApp()
      .then(function() {
        replaceFile('.ember-cli', '"disableAnalytics": false', '"disableAnalytics": false,' + EOL + '"usePods" : true' + EOL);
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

  function destroyAfterGenerateWithPodsByDefault(args) {
    return initApp()
      .then(function() {
        replaceFile('config/environment.js', "var ENV = {", "var ENV = {" + EOL + "usePodsByDefault: true, " + EOL);
        return generate(args);
      })
      .then(function() {
        return destroy(args);
      });
  }

  function destroyAfterGenerate(args) {
    return initApp()
      .then(function() {
        replaceFile('config/environment.js', "var ENV = {", "var ENV = {" + EOL + "podModulePrefix: 'app/pods', " + EOL);
        return generate(args);
      })
      .then(function() {
        return destroy(args);
      });
  }

  it('.ember-cli usePods setting destroys in pod structure without --pod flag', function() {
    var commandArgs = ['controller', 'foo'];
    var files       = [
      'app/foo/controller.js',
      'tests/unit/foo/controller-test.js'
    ];

    return assertDestroyAfterGenerateWithUsePods(commandArgs, files);
  });

  it('.ember-cli usePods setting destroys in classic structure with --classic flag', function() {
    var commandArgs = ['controller', 'foo', '--classic'];
    var files       = [
      'app/controllers/foo.js',
      'tests/unit/controllers/foo-test.js'
    ];

    return assertDestroyAfterGenerateWithUsePods(commandArgs, files);
  });

  it('.ember-cli usePods setting correctly destroys component', function() {
    var commandArgs = ['component', 'x-foo'];
    var files       = [
      'app/components/x-foo/component.js',
      'app/components/x-foo/template.hbs',
      'tests/integration/components/x-foo/component-test.js'
    ];

    return assertDestroyAfterGenerateWithUsePods(commandArgs, files);
  });

  it('controller foo --pod', function() {
    var commandArgs = ['controller', 'foo', '--pod'];
    var files       = [
      'app/pods/foo/controller.js',
      'tests/unit/pods/foo/controller-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('controller foo/bar --pod', function() {
    var commandArgs = ['controller', 'foo/bar', '--pod'];
    var files       = [
      'app/pods/foo/bar/controller.js',
      'tests/unit/pods/foo/bar/controller-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('component x-foo --pod', function() {
    var commandArgs = ['component', 'x-foo', '--pod'];
    var files       = [
      'app/pods/components/x-foo/component.js',
      'app/pods/components/x-foo/template.hbs',
      'tests/integration/pods/components/x-foo/component-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('helper foo-bar --pod', function() {
    var commandArgs = ['helper', 'foo-bar', '--pod'];
    var files       = [
      'app/helpers/foo-bar.js',
      'tests/unit/helpers/foo-bar-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('helper foo/bar-baz --pod', function() {
    var commandArgs = ['helper', 'foo/bar-baz', '--pod'];
    var files       = [
      'app/helpers/foo/bar-baz.js',
      'tests/unit/helpers/foo/bar-baz-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('model foo --pod', function() {
    var commandArgs = ['model', 'foo', '--pod'];
    var files       = [
      'app/pods/foo/model.js',
      'tests/unit/pods/foo/model-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('model foo/bar --pod', function() {
    var commandArgs = ['model', 'foo/bar', '--pod'];
    var files       = [
      'app/pods/foo/bar/model.js',
      'tests/unit/pods/foo/bar/model-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('route foo --pod', function() {
    var commandArgs = ['route', 'foo', '--pod'];
    var files       = [
      'app/pods/foo/route.js',
      'app/pods/foo/template.hbs',
      'tests/unit/pods/foo/route-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files)
      .then(function() {
        assertFile('app/router.js', {
          doesNotContain: "this.route('foo');"
        });
      });
  });

  it('route index --pod', function() {
    var commandArgs = ['route', 'index', '--pod'];
    var files       = [
      'app/pods/index/route.js',
      'app/pods/index/template.hbs',
      'tests/unit/pods/index/route-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('route basic --pod', function() {
    var commandArgs = ['route', 'basic', '--pod'];
    var files       = [
      'app/pods/basic/route.js',
      'app/pods/basic/template.hbs',
      'tests/unit/pods/basic/route-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('resource foo --pod', function() {
    var commandArgs = ['resource', 'foo', '--pod'];
    var files       = [
      'app/pods/foo/model.js',
      'tests/unit/pods/foo/model-test.js',
      'app/pods/foo/route.js',
      'tests/unit/pods/foo/route-test.js',
      'app/pods/foo/template.hbs'
    ];

    return assertDestroyAfterGenerate(commandArgs, files)
      .then(function() {
        assertFile('app/router.js', {
          doesNotContain: "this.route('foo');"
        });
      });
  });

  it('template foo --pod', function() {
    var commandArgs = ['template', 'foo', '--pod'];
    var files       = ['app/pods/foo/template.hbs'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('template foo/bar --pod', function() {
    var commandArgs = ['template', 'foo/bar', '--pod'];
    var files       = ['app/pods/foo/bar/template.hbs'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('view foo --pod', function() {
    var commandArgs = ['view', 'foo', '--pod'];
    var files       = [
      'app/pods/foo/view.js',
      'tests/unit/pods/foo/view-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('view foo/bar --pod', function() {
    var commandArgs = ['view', 'foo/bar', '--pod'];
    var files       = [
      'app/pods/foo/bar/view.js',
      'tests/unit/pods/foo/bar/view-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('initializer foo --pod', function() {
    var commandArgs = ['initializer', 'foo', '--pod'];
    var files       = ['app/initializers/foo.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('initializer foo/bar', function() {
    var commandArgs = ['initializer', 'foo/bar', '--pod'];
    var files       = ['app/initializers/foo/bar.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('mixin foo --pod', function() {
    var commandArgs = ['mixin', 'foo', '--pod'];
    var files       = [
      'app/mixins/foo.js',
      'tests/unit/mixins/foo-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('mixin foo/bar --pod', function() {
    var commandArgs = ['mixin', 'foo/bar', '--pod'];
    var files       = [
      'app/mixins/foo/bar.js',
      'tests/unit/mixins/foo/bar-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('adapter foo --pod', function() {
    var commandArgs = ['adapter', 'foo', '--pod'];
    var files       = ['app/pods/foo/adapter.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('adapter foo/bar --pod', function() {
    var commandArgs = ['adapter', 'foo/bar', '--pod'];
    var files       = ['app/pods/foo/bar/adapter.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('serializer foo --pod', function() {
    var commandArgs = ['serializer', 'foo', '--pod'];
    var files       = [
      'app/pods/foo/serializer.js',
      'tests/unit/pods/foo/serializer-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('serializer foo/bar --pod', function() {
    var commandArgs = ['serializer', 'foo/bar', '--pod'];
    var files       = [
      'app/pods/foo/bar/serializer.js',
      'tests/unit/pods/foo/bar/serializer-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('transform foo --pod', function() {
    var commandArgs = ['transform', 'foo', '--pod'];
    var files       = [
      'app/pods/foo/transform.js',
      'tests/unit/pods/foo/transform-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('transform foo/bar --pod', function() {
    var commandArgs = ['transform', 'foo/bar', '--pod'];
    var files       = [
      'app/pods/foo/bar/transform.js',
      'tests/unit/pods/foo/bar/transform-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('util foo-bar --pod', function() {
    var commandArgs = ['util', 'foo-bar', '--pod'];
    var files       = [
      'app/utils/foo-bar.js',
      'tests/unit/utils/foo-bar-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('util foo-bar/baz --pod', function() {
    var commandArgs = ['util', 'foo/bar-baz', '--pod'];
    var files       = [
      'app/utils/foo/bar-baz.js',
      'tests/unit/utils/foo/bar-baz-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('service foo --pod', function() {
    var commandArgs = ['service', 'foo', '--pod'];
    var files       = [
      'app/pods/foo/service.js',
      'tests/unit/pods/foo/service-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('service foo/bar --pod', function() {
    var commandArgs = ['service', 'foo/bar', '--pod'];
    var files       = [
      'app/pods/foo/bar/service.js',
      'tests/unit/pods/foo/bar/service-test.js'
    ];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('blueprint foo --pod', function() {
    var commandArgs = ['blueprint', 'foo', '--pod'];
    var files       = ['blueprints/foo/index.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('blueprint foo/bar --pod', function() {
    var commandArgs = ['blueprint', 'foo/bar', '--pod'];
    var files       = ['blueprints/foo/bar/index.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('http-mock foo --pod', function() {
    var commandArgs = ['http-mock', 'foo', '--pod'];
    var files       = ['server/mocks/foo.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('http-proxy foo --pod', function() {
    var commandArgs = ['http-proxy', 'foo', 'bar', '--pod'];
    var files       = ['server/proxies/foo.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('in-addon component x-foo --pod', function() {
    var commandArgs = ['component', 'x-foo', '--pod'];
    var files       = [
      'addon/components/x-foo/component.js',
      'addon/components/x-foo/template.hbs',
      'app/components/x-foo/component.js',
      'tests/integration/components/x-foo/component-test.js'
    ];

    return assertDestroyAfterGenerateInAddon(commandArgs, files);
  });

  it('in-repo-addon component x-foo --pod', function(){
    var commandArgs = ['component', 'x-foo', '--in-repo-addon=my-addon', '--pod'];
    var files       = [
      'lib/my-addon/addon/components/x-foo/component.js',
      'lib/my-addon/addon/components/x-foo/template.hbs',
      'lib/my-addon/app/components/x-foo/component.js',
      'tests/integration/components/x-foo/component-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('in-repo-addon component nested/x-foo --pod', function(){
    var commandArgs = ['component', 'nested/x-foo', '--in-repo-addon=my-addon', '--pod'];
    var files       = [
      'lib/my-addon/addon/components/nested/x-foo/component.js',
      'lib/my-addon/addon/components/nested/x-foo/template.hbs',
      'lib/my-addon/app/components/nested/x-foo/component.js',
      'tests/integration/components/nested/x-foo/component-test.js'
    ];

    return assertDestroyAfterGenerateInRepoAddon(commandArgs, files);
  });

  it('acceptance-test foo --pod', function() {
    var commandArgs = ['acceptance-test', 'foo', '--pod'];
    var files       = ['tests/acceptance/foo-test.js'];

    return assertDestroyAfterGenerate(commandArgs, files);
  });

  it('deletes files generated using blueprints from the project directory', function() {
    var commandArgs = ['foo', 'bar', '--pod'];
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
    var commandArgs = ['controller', 'foo', '--pod'];
    var files       = ['app/foo/controller.js'];
    return initApp()
      .then(function() {
        return outputFile(
          'blueprints/controller/files/app/__path__/__name__.js',
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

  // Skip until podModulePrefix is deprecated
  it.skip('podModulePrefix deprecation warning', function() {
    return destroyAfterGenerate(['controller', 'foo', '--pod']).then(function(result) {
      expect(result.ui.output).to.include("`podModulePrefix` is deprecated and will be"+
      " removed from future versions of ember-cli. Please move existing pods from"+
      " 'app/pods/' to 'app/'.");
    });
  });

  it('usePodsByDefault deprecation warning', function() {
    return destroyAfterGenerateWithPodsByDefault(['controller', 'foo', '--pod']).then(function(result) {
      expect(result.ui.output).to.include('`usePodsByDefault` is no longer supported in'+
        ' \'config/environment.js\', use `usePods` in \'.ember-cli\' instead.');
    });
  });

});
