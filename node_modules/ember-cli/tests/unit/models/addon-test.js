'use strict';

var fs      = require('fs-extra');
var path    = require('path');
var Project = require('../../../lib/models/project');
var Addon   = require('../../../lib/models/addon');
var Promise = require('../../../lib/ext/promise');
var expect  = require('chai').expect;
var remove  = Promise.denodeify(fs.remove);
var tmp     = require('tmp-sync');
var path    = require('path');
var findWhere = require('lodash/collection/find');
var MockUI = require('../../helpers/mock-ui');

var broccoli  = require('broccoli');
var walkSync  = require('walk-sync');

var root    = process.cwd();
var tmproot = path.join(root, 'tmp');

var fixturePath = path.resolve(__dirname, '../../fixtures/addon');

describe('models/addon.js', function() {
  var addon, project, projectPath;

  describe('root property', function() {
    it('is required', function() {
      expect(function() {
        var TheAddon = Addon.extend({root:undefined});
        new TheAddon();
      }).to.throw(/root/);
    });
  });

  describe('treePaths and treeForMethods', function() {
    var FirstAddon, SecondAddon;

    beforeEach(function() {
      projectPath = path.resolve(fixturePath, 'simple');
      var packageContents = require(path.join(projectPath, 'package.json'));

      project = new Project(projectPath, packageContents);

      FirstAddon = Addon.extend({
        name: 'first',
        root: projectPath,

        init: function() {
          this.treePaths.vendor = 'blazorz';
          this.treeForMethods.public = 'huzzah!';
        }
      });

      SecondAddon = Addon.extend({
        name: 'first',
        root: projectPath,

        init: function() {
          this.treePaths.vendor = 'blammo';
          this.treeForMethods.public = 'boooo';
        }
      });

    });

    describe('.jshintAddonTree', function() {
      var addon;

      beforeEach(function() {
        addon = new FirstAddon(project);

        // TODO: fix config story...
        addon.app = {
          options: { jshintrc: {} },
          addonLintTree: function(type, tree) { return tree; }
        };

        addon.jshintTrees = function(){};

      });

      it('uses the fullPath', function() {
        var addonPath;
        addon.addonJsFiles = function(_path) {
          addonPath = _path;
          return _path;
        };

        var root = path.join(fixturePath, 'with-styles');
        addon.root = root;

        addon.jshintAddonTree();
        expect(addonPath).to.eql(path.join(root, 'addon'));
      });

      it('lints the files before preprocessing', function() {
        addon.preprocessJs = function() {
          expect(false, 'should not preprocess files').to.eql(true);
        };

        var root = path.join(fixturePath, 'with-styles');
        addon.root = root;

        addon.jshintAddonTree();
      });

    });

    it('modifying a treePath does not affect other addons', function() {
      var first = new FirstAddon(project);
      var second = new SecondAddon(project);

      expect(first.treePaths.vendor).to.equal('blazorz');
      expect(second.treePaths.vendor).to.equal('blammo');
    });

    it('modifying a treeForMethod does not affect other addons', function() {
      var first = new FirstAddon(project);
      var second = new SecondAddon(project);

      expect(first.treeForMethods.public).to.equal('huzzah!');
      expect(second.treeForMethods.public).to.equal('boooo');
    });
  });

  describe('resolvePath', function() {
    beforeEach(function() {
      addon = {
        pkg: {
          'ember-addon': {
            'main': ''
          }
        },
        path: ''
      };
    });

    it('adds .js if not present', function() {
      addon.pkg['ember-addon']['main'] = 'index';
      var resolvedFile = path.basename(Addon.resolvePath(addon));
      expect(resolvedFile).to.equal('index.js');
    });

    it('doesn\'t add .js if it is .js', function() {
      addon.pkg['ember-addon']['main'] = 'index.js';
      var resolvedFile = path.basename(Addon.resolvePath(addon));
      expect(resolvedFile).to.equal('index.js');
    });

    it('doesn\'t add .js if it has another extension', function() {
      addon.pkg['ember-addon']['main'] = 'index.coffee';
      var resolvedFile = path.basename(Addon.resolvePath(addon));
      expect(resolvedFile).to.equal('index.coffee');
    });

    it('allows lookup of non-`index.js` `main` entry points', function() {
      delete addon.pkg['ember-addon'];
      addon.pkg['main'] = 'some/other/path.js';

      var resolvedFile = Addon.resolvePath(addon);
      expect(resolvedFile).to.equal(path.join(process.cwd(), 'some/other/path.js'));
    });

    it('falls back to `index.js` if `main` and `ember-addon` are not found', function() {
      delete addon.pkg['ember-addon'];

      var resolvedFile = Addon.resolvePath(addon);
      expect(resolvedFile).to.equal(path.join(process.cwd(), 'index.js'));
    });

    it('falls back to `index.js` if `main` and `ember-addon.main` are not found', function() {
      delete addon.pkg['ember-addon'].main;

      var resolvedFile = Addon.resolvePath(addon);
      expect(resolvedFile).to.equal(path.join(process.cwd(), 'index.js'));
    });
  });

  describe('initialized addon', function() {
    this.timeout(40000);
    before(function() {
      projectPath = path.resolve(fixturePath, 'simple');
      var packageContents = require(path.join(projectPath, 'package.json'));
      project = new Project(projectPath, packageContents);
      project.initializeAddons();
    });

    describe('generated addon', function() {
      beforeEach(function() {
        addon = findWhere(project.addons, { name: 'Ember CLI Generated with export' });

        // Clear the caches
        delete addon._moduleName;
      });

      it('sets it\'s project', function() {
        expect(addon.project.name).to.equal(project.name);
      });

      it('sets it\'s parent', function() {
        expect(addon.parent.name).to.equal(project.name);
      });

      it('sets the root', function() {
        expect(addon.root).to.not.equal(undefined);
      });

      it('sets the pkg', function() {
        expect(addon.pkg).to.not.equal(undefined);
      });

      describe('trees for it\'s treePaths', function() {
        it('app', function() {
          var tree = addon.treeFor('app');
          expect(typeof (tree.read || tree.rebuild)).to.equal('function');
        });

        it('styles', function() {
          var tree = addon.treeFor('styles');
          expect(typeof (tree.read || tree.rebuild)).to.equal('function');
        });

        it('templates', function() {
          var tree = addon.treeFor('templates');
          expect(typeof (tree.read || tree.rebuild)).to.equal('function');
        });

        it('addon-templates', function() {
          var tree = addon.treeFor('addon-templates');
          expect(typeof (tree.read || tree.rebuild)).to.equal('function');
        });

        it('vendor', function() {
          var tree = addon.treeFor('vendor');
          expect(typeof (tree.read || tree.rebuild)).to.equal('function');
        });

        it('addon', function() {
          var app = {
            importWhitelist: {},
            options: {},
          };
          addon.registry = {
            app: addon,
            load: function() {
              return [{
                toTree: function(tree) {
                  return tree;
                }
              }];
            },

            extensionsForType: function() {
              return ['js'];
            }
          };
          addon.app = app;
          var tree = addon.treeFor('addon');
          expect(typeof (tree.read || tree.rebuild)).to.equal('function');
        });
      });

      describe('custom treeFor methods', function() {
        it('can define treeForApp', function() {
          var called = false;

          addon.treeForApp = function() {
            called = true;
          };

          addon.treeFor('app');
          expect(called).to.equal(true);
        });

        it('can define treeForStyles', function() {
          var called = false;

          addon.treeForStyles = function() {
            called = true;
          };

          addon.treeFor('styles');
          expect(called).to.equal(true);
        });

        it('can define treeForVendor', function() {
          var called = false;

          addon.treeForVendor = function() {
            called = true;
          };

          addon.treeFor('vendor');
          expect(called).to.equal(true);
        });

        it('can define treeForTemplates', function() {
          var called = false;

          addon.treeForTemplates = function() {
            called = true;
          };

          addon.treeFor('templates');
          expect(called).to.equal(true);
        });

        it('can define treeForAddonTemplates', function() {
          var called = false;

          addon.treeForAddonTemplates = function() {
            called = true;
          };

          addon.treeFor('addon-templates');
          expect(called).to.equal(true);
        });

        it('can define treeForPublic', function() {
          var called = false;

          addon.treeForPublic = function() {
            called = true;
          };

          addon.treeFor('public');
          expect(called).to.equal(true);
        });
      });
    });

    describe('addon with dependencies', function() {
      beforeEach(function() {
        addon = findWhere(project.addons, { name: 'Ember Addon With Dependencies' });
      });

      it('returns a listing of all dependencies in the addon\'s package.json', function() {
        var expected = {
          'ember-cli': 'latest',
          'something-else': 'latest'
        };

        expect(addon.dependencies()).to.deep.equal(expected);
      });
    });

    it('must define a `name` property', function() {
      var Foo = Addon.extend({ root: 'foo' });

      expect(function() {
        new Foo(project);
      }).to.throw(/An addon must define a `name` property./);
    });

    describe('isDevelopingAddon', function() {
      var originalEnvValue, addon, project;

      beforeEach(function() {
        var MyAddon = Addon.extend({
          name: 'test-project',
          root: 'foo'
        });

        var projectPath = path.resolve(fixturePath, 'simple');
        var packageContents = require(path.join(projectPath, 'package.json'));

        project = new Project(projectPath, packageContents);

        addon = new MyAddon(project);

        originalEnvValue = process.env.EMBER_ADDON_ENV;
      });

      afterEach(function() {
        if(originalEnvValue === undefined) {
          delete process.env.EMBER_ADDON_ENV;
        } else {
          process.env.EMBER_ADDON_ENV = originalEnvValue;
        }
      });

      it('returns true when `EMBER_ADDON_ENV` is set to development', function() {
        process.env.EMBER_ADDON_ENV = 'development';

        expect(addon.isDevelopingAddon(), 'addon is being developed').to.eql(true);
      });

      it('returns false when `EMBER_ADDON_ENV` is not set', function() {
        delete process.env.EMBER_ADDON_ENV;

        expect(addon.isDevelopingAddon()).to.eql(false);
      });

      it('returns false when `EMBER_ADDON_ENV` is something other than `development`', function() {
        process.env.EMBER_ADDON_ENV = 'production';

        expect(addon.isDevelopingAddon()).to.equal(false);
      });

      it('returns false when the addon is not the one being developed', function() {
        process.env.EMBER_ADDON_ENV = 'development';

        addon.name = 'my-addon';
        expect(addon.isDevelopingAddon(), 'addon is not being developed').to.eql(false);
      });
    });

    describe('treeGenerator', function() {
      it('watch tree when developing the addon itself', function() {
        addon.isDevelopingAddon = function() { return true; };

        var tree = addon.treeGenerator('foo/bar');

        expect(tree.__broccoliGetInfo__()).to.have.property('watched', true);
      });

      it('uses UnwatchedDir when not developing the addon itself', function() {
        addon.isDevelopingAddon = function() { return false; };

        var tree = addon.treeGenerator('foo/bar');

        expect(tree.__broccoliGetInfo__()).to.have.property('watched', false);
      });
    });

    describe('blueprintsPath', function() {
      var tmpdir;

      beforeEach(function() {
        tmpdir  = tmp.in(tmproot);

        addon.root = tmpdir;
      });

      afterEach(function() {
        return remove(tmproot);
      });

      it('returns undefined if the `blueprint` folder does not exist', function() {
        var returnedPath = addon.blueprintsPath();

        expect(returnedPath).to.equal(undefined);
      });

      it('returns blueprint path if the folder exists', function() {
        var blueprintsDir = path.join(tmpdir, 'blueprints');
        fs.mkdirSync(blueprintsDir);

        var returnedPath = addon.blueprintsPath();

        expect(returnedPath).to.equal(blueprintsDir);
      });
    });

    describe('config', function() {
      it('returns undefined if `config/environment.js` does not exist', function() {
        addon.root = path.join(fixturePath, 'no-config');
        var result = addon.config();

        expect(result).to.equal(undefined);
      });

      it('returns blueprint path if the folder exists', function() {
        addon.root = path.join(fixturePath, 'with-config');
        var appConfig = {};

        addon.config('development', appConfig);

        expect(appConfig.addon).to.equal('with-config');
      });
    });
  });

  describe('Addon.lookup', function() {
    it('should throw an error if an addon could not be found', function() {
      var addon = {
        path: 'foo/bar-baz/blah/doesnt-exist',
        pkg: {
          name: 'dummy-addon',
          'ember-addon': { }
        }
      };

      expect(function() {
        Addon.lookup(addon);
      }).to.throw(/The `dummy-addon` addon could not be found at `foo\/bar-baz\/blah\/doesnt-exist`\./);
    });
  });

  describe('compileTemplates', function() {
    beforeEach(function() {
      projectPath = path.resolve(fixturePath, 'simple');
      var packageContents = require(path.join(projectPath, 'package.json'));

      project = new Project(projectPath, packageContents);

      project.initializeAddons();

      addon = findWhere(project.addons, { name: 'Ember CLI Generated with export' });
    });

    it('should throw a useful error if a template compiler is not present -- non-pods', function() {
      addon.root = path.join(fixturePath, 'with-addon-templates');

      expect(function() {
        addon.compileTemplates();
      }).to.throw(
        'Addon templates were detected, but there ' +
        'are no template compilers registered for `' + addon.name + '`. ' +
        'Please make sure your template precompiler (commonly `ember-cli-htmlbars`) ' +
        'is listed in `dependencies` (NOT `devDependencies`) in ' +
        '`' + addon.name + '`\'s `package.json`.'
      );
    });

    it('should throw a useful error if a template compiler is not present -- pods', function() {
      addon.root = path.join(fixturePath, 'with-addon-pod-templates');

      expect(function() {
        addon.compileTemplates();
      }).to.throw(
        'Addon templates were detected, but there ' +
        'are no template compilers registered for `' + addon.name + '`. ' +
        'Please make sure your template precompiler (commonly `ember-cli-htmlbars`) ' +
        'is listed in `dependencies` (NOT `devDependencies`) in ' +
        '`' + addon.name + '`\'s `package.json`.'
      );
    });

    it('should not throw an error if addon/templates is present but empty', function() {
      addon.root = path.join(fixturePath, 'with-empty-addon-templates');

      expect(function() {
        addon.compileTemplates();
      }).not.to.throw();
    });
  });

  describe('addonDiscovery', function() {
    var discovery, addon, ui;

    beforeEach(function() {
      projectPath = path.resolve(fixturePath, 'simple');
      var packageContents = require(path.join(projectPath, 'package.json'));

      ui = new MockUI();
      project = new Project(projectPath, packageContents, ui);

      var AddonTemp = Addon.extend({
        name: 'temp',
        root: 'foo'
      });

      addon = new AddonTemp(project, project);
      discovery = addon.addonDiscovery;
    });

    it('is provided with the addon\'s `ui` object', function() {
      expect(discovery.ui).to.equal(ui);
    });
  });

  describe('treeForStyles', function() {
    var builder, addon;

    beforeEach(function() {
      projectPath = path.resolve(fixturePath, 'with-app-styles');
      var packageContents = require(path.join(projectPath, 'package.json'));

      project = new Project(projectPath, packageContents);

      var BaseAddon = Addon.extend({
        name: 'base-addon',
        root: projectPath
      });

      addon = new BaseAddon(project, project);
    });

    afterEach(function() {
      if (builder) {
        return builder.cleanup();
      }
    });

    it('should move files in the root of the addons app/styles tree into the app/styles path', function() {
      builder = new broccoli.Builder(addon.treeFor('styles'));

      return builder.build()
        .then(function(results) {
          var outputPath = results.directory;

          var expected = [
            'app/',
            'app/styles/',
            'app/styles/foo-bar.css'
          ];

          expect(walkSync(outputPath)).to.eql(expected);
        });
    });
  });
});
