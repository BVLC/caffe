/* global escape */

'use strict';

var fs         = require('fs');
var path       = require('path');
var Project    = require('../../../lib/models/project');
var expect     = require('chai').expect;
var stub       = require('../../helpers/stub').stub;
var proxyquire = require('proxyquire');

var mergeTreesStub;
var EmberApp = proxyquire('../../../lib/broccoli/ember-app', {
  './merge-trees': function() {
    return mergeTreesStub.apply(this, arguments);
  }
});

describe('broccoli/ember-app', function() {
  var project, projectPath, emberApp, addonTreesForStub, addon;

  function setupProject(rootPath) {
    var packageContents = require(path.join(rootPath, 'package.json'));

    project = new Project(rootPath, packageContents);
    project.require = function() {
      return function() {};
    };
    project.initializeAddons = function() {
      this.addons = [];
    };

    return project;
  }

  beforeEach(function() {
    projectPath = path.resolve(__dirname, '../../fixtures/addon/simple');
    project = setupProject(projectPath);

    mergeTreesStub = require('../../../lib/broccoli/merge-trees');
  });

  describe('constructor', function() {
    it('should override project.configPath if configPath option is specified', function() {
      project.configPath = function() { return 'original value'; };

      new EmberApp({
        project: project,
        configPath: 'custom config path'
      });

      expect(project.configPath()).to.equal('custom config path');
    });

    it('should set bowerDirectory for app', function() {
      var app = new EmberApp({
        project: project
      });

      expect(app.bowerDirectory).to.equal(project.bowerDirectory);
      expect(app.bowerDirectory).to.equal('bower_components');
    });

    it('should merge options with defaults to depth', function() {
      var app = new EmberApp({
        project: project,
        foo: {
          bar: ['baz']
        },
        fooz: {
          bam: {
            boo: ['default']
          }
        }
      }, {
        foo: {
          bar: ['bizz']
        },
        fizz: 'fizz',
        fooz: {
          bam: {
            boo: ['custom']
          }
        }
      });

      expect(app.options.foo).to.deep.eql({
        bar: ['bizz']
      });
      expect(app.options.fizz).to.eql('fizz');
      expect(app.options.fooz).to.eql({
        bam: {
          boo: ['custom']
        }
      });
    });

    it('should do the right thing when merging default object options', function() {
      var app = new EmberApp({
        project: project,
      }, {
        minifyJS: {
          enabled: true,
          options: {
            exclusions: ['hey', 'you']
          }
        }
      });

      expect(app.options.minifyJS).to.deep.equal({
        enabled: true,
        options: {
          exclusions: ['hey', 'you']
        }
      });
    });

    describe('_notifyAddonIncluded', function() {
      beforeEach(function() {
        project.initializeAddons = function() { };
        project.addons = [{name: 'custom-addon'}];
      });

      it('should set the app on the addons', function() {
        var app = new EmberApp({
          project: project
        });

        var addon = project.addons[0];
        expect(addon.app).to.deep.equal(app);
      });
    });
  });

  describe('contentFor', function() {
    var config, defaultMatch;

    beforeEach(function() {
      project._addonsInitialized = true;
      project.addons = [];

      emberApp = new EmberApp({
        project: project
      });

      config = {
        modulePrefix: 'cool-foo'
      };

      defaultMatch = '{{content-for \'head\'}}';
    });

    describe('contentFor from addons', function() {
      it('calls `contentFor` on addon', function() {
        var calledConfig, calledType;

        project.addons.push({
          contentFor: function(type, config) {
            calledType = type;
            calledConfig = config;

            return 'blammo';
          }
        });

        var actual = emberApp.contentFor(config, defaultMatch, 'foo');

        expect(calledConfig).to.deep.equal(config);
        expect(calledType).to.equal('foo');
        expect(actual).to.equal('blammo');
      });

      it('calls `contentFor` on each addon', function() {
        project.addons.push({
          contentFor: function() {
            return 'blammo';
          }
        });

        project.addons.push({
          contentFor: function() {
            return 'blahzorz';
          }
        });

        var actual = emberApp.contentFor(config, defaultMatch, 'foo');

        expect(actual).to.equal('blammo\nblahzorz');
      });

      it('allows later addons to inspect previous content', function() {
        var calledContent;

        project.addons.push({
          contentFor: function() {
            return 'zero';
          }
        });

        project.addons.push({
          contentFor: function() {
            return 'one';
          }
        });

        project.addons.push({
          contentFor: function(type, config, content) {
            calledContent = content.slice();
            content.pop();
            return 'two';
          }
        });

        var actual = emberApp.contentFor(config, defaultMatch, 'foo');

        expect(calledContent).to.deep.equal(['zero', 'one']);
        expect(actual).to.equal('zero\ntwo');
      });
    });

    describe('contentFor("head")', function() {
      it('includes the `meta` tag in `head` by default', function() {
        var escapedConfig = escape(JSON.stringify(config));
        var metaExpected = '<meta name="cool-foo/config/environment" ' +
                           'content="' + escapedConfig + '" />';
        var actual = emberApp.contentFor(config, defaultMatch, 'head');

        expect(true, actual.indexOf(metaExpected) > -1);
      });

      it('does not include the `meta` tag in `head` if storeConfigInMeta is false', function() {
        emberApp.options.storeConfigInMeta = false;

        var escapedConfig = escape(JSON.stringify(config));
        var metaExpected = '<meta name="cool-foo/config/environment" ' +
                           'content="' + escapedConfig + '" />';
        var actual = emberApp.contentFor(config, defaultMatch, 'head');

        expect(true, actual.indexOf(metaExpected) === -1);
      });

      it('includes the `base` tag in `head` if locationType is auto', function() {
        config.locationType = 'auto';
        config.baseURL = '/';
        var expected = '<base href="/" />';
        var actual = emberApp.contentFor(config, defaultMatch, 'head');

        expect(true, actual.indexOf(expected) > -1);
      });

      it('includes the `base` tag in `head` if locationType is none (testem requirement)', function() {
        config.locationType = 'none';
        config.baseURL = '/';
        var expected = '<base href="/" />';
        var actual = emberApp.contentFor(config, defaultMatch, 'head');

        expect(true, actual.indexOf(expected) > -1);
      });

      it('does not include the `base` tag in `head` if locationType is hash', function() {
        config.locationType = 'hash';
        config.baseURL = '/foo/bar';
        var expected = '<base href="/foo/bar/" />';
        var actual = emberApp.contentFor(config, defaultMatch, 'head');

        expect(true, actual.indexOf(expected) === -1);
      });
    });

    describe('contentFor("config-module")', function() {
      it('includes the meta gathering snippet by default', function() {
        var metaSnippetPath = path.join(__dirname, '..','..','..','lib','broccoli','app-config-from-meta.js');
        var expected = fs.readFileSync(metaSnippetPath);

        var actual = emberApp.contentFor(config, defaultMatch, 'config-module');

        expect(true, actual.indexOf(expected) > -1);
      });

      it('includes the raw config if storeConfigInMeta is false', function() {
        emberApp.options.storeConfigInMeta = false;

        var expected = JSON.stringify(config);
        var actual = emberApp.contentFor(config, defaultMatch, 'config-module');

        expect(true, actual.indexOf(expected) > -1);
      });
    });

    it('has no default value other than `head`', function() {
      expect(emberApp.contentFor(config, defaultMatch, 'foo')).to.equal('');
      expect(emberApp.contentFor(config, defaultMatch, 'body')).to.equal('');
      expect(emberApp.contentFor(config, defaultMatch, 'blah')).to.equal('');
    });
  });

  describe('addons', function() {
    describe('included hook', function() {
      it('included hook is called properly on instantiation', function() {
        var called = false;
        var passedApp;

        addon = {
          included: function(app) { called = true; passedApp = app; },
          treeFor: function() { }
        };

        project.initializeAddons = function() {
          this.addons = [ addon ];
        };

        emberApp = new EmberApp({
          project: project
        });

        expect(true, called);
        expect(passedApp).to.equal(emberApp);
      });

      it('does not throw an error if the addon does not implement `included`', function() {
        delete addon.included;

        project.initializeAddons = function() {
          this.addons = [ addon ];
        };

        expect(function() {
          emberApp = new EmberApp({
            project: project
          });
        }).to.not.throw(/addon must implement the `included`/);
      });
    });

    describe('addonTreesFor', function() {
      beforeEach(function() {
        addon = {
          included: function() { },
          treeFor: function() { }
        };

        project.initializeAddons = function() {
          this.addons = [ addon ];
        };

      });

      it('addonTreesFor returns an empty array if no addons return a tree', function() {
        emberApp = new EmberApp({
          project: project
        });

        expect(emberApp.addonTreesFor('blah')).to.deep.equal([]);
      });

      it('addonTreesFor calls treesFor on the addon', function() {
        emberApp = new EmberApp({
          project: project
        });

        var sampleAddon = project.addons[0];
        var actualTreeName;

        sampleAddon.treeFor = function(name) {
          actualTreeName = name;

          return 'blazorz';
        };

        expect(emberApp.addonTreesFor('blah')).to.deep.equal(['blazorz']);
        expect(actualTreeName).to.equal('blah');
      });

      it('addonTreesFor does not throw an error if treeFor is not defined', function() {
        delete addon.treeFor;

        emberApp = new EmberApp({
          project: project
        });

        expect(function() {
          emberApp.addonTreesFor('blah');
        }).not.to.throw(/addon must implement the `treeFor`/);
      });

      describe('addonTreesFor is called properly', function() {
        beforeEach(function() {
          emberApp = new EmberApp({
            project: project
          });

          addonTreesForStub = stub(emberApp, 'addonTreesFor', ['batman']);
        });

        it('_processedVendorTree calls addonTreesFor', function() {
          emberApp._processedVendorTree();

          expect(addonTreesForStub.calledWith[0][0]).to.equal('addon');
          expect(addonTreesForStub.calledWith[1][0]).to.equal('vendor');
        });

        it('_processedAppTree calls addonTreesFor', function() {
          emberApp._processedAppTree();

          expect(addonTreesForStub.calledWith[0][0]).to.equal('app');
        });
      });
    });
    describe('postprocessTree is called properly', function() {
        var postprocessTreeStub;
        beforeEach(function() {
          emberApp = new EmberApp({
            project: project
          });

          postprocessTreeStub = stub(emberApp, 'addonPostprocessTree', ['batman']);
        });


        it('styles calls addonTreesFor', function() {
          emberApp.styles();

          expect(postprocessTreeStub.calledWith[0][0]).to.equal('css');
          expect(postprocessTreeStub.calledWith[0][1].description).to.equal('styles', 'should be called with consolidated tree');
        });


        it('template type is called', function() {
          var oldLoad = emberApp.registry.load;
          emberApp.registry.load = function(type) {
            if (type === 'template'){
              return [
                {
                  toTree: function() {
                    return {
                      description: 'template'
                    };
                  }
                }];
            } else {
              return oldLoad.call(emberApp.registry, type);
            }
          };

          emberApp._processedTemplatesTree();
          expect(postprocessTreeStub.calledWith[0][0]).to.equal('template');
          expect(postprocessTreeStub.calledWith[0][1].description).to.equal('template', 'should be called with consolidated tree');
        });
    });

    describe('toTree', function() {
      beforeEach(function() {
        addon = {
          included: function() { },
          treeFor: function() { },
          postprocessTree: function() { }
        };

        project.initializeAddons = function() {
          this.addons = [ addon ];
        };

        emberApp = new EmberApp({
          project: project
        });
      });

      it('calls postProcessTree if defined', function() {
        stub(emberApp, 'toArray', []);
        stub(addon, 'postprocessTree', 'derp');

        expect(emberApp.toTree()).to.equal('derp');
      });

      it('calls addonPostprocessTree', function() {
        stub(emberApp, 'toArray', []);
        stub(emberApp, 'addonPostprocessTree', 'blap');

        expect(emberApp.toTree()).to.equal('blap');
      });

      it('calls each addon postprocessTree hook', function() {
        stub(emberApp, '_processedTemplatesTree', 'x');
        stub(addon, 'postprocessTree', 'blap');
        expect(emberApp.toTree()).to.equal('blap');
        expect(
          addon.postprocessTree.calledWith.map(function(args){
            return args[0];
          }).sort()
        ).to.deep.equal(['all', 'css', 'js', 'test']);
      });

    });

    describe('isEnabled is called properly', function() {
      beforeEach(function() {
        projectPath = path.resolve(__dirname, '../../fixtures/addon/env-addons');
        var packageContents = require(path.join(projectPath, 'package.json'));
        project = new Project(projectPath, packageContents);
      });

      afterEach(function() {
        process.env.EMBER_ENV = undefined;
      });

      describe('with environment', function() {
        it('development', function() {
          process.env.EMBER_ENV = 'development';
          emberApp = new EmberApp({ project: project });

          expect(emberApp.project.addons.length).to.equal(5);
        });

        it('foo', function() {
          process.env.EMBER_ENV = 'foo';
          emberApp = new EmberApp({ project: project });

          expect(emberApp.project.addons.length).to.equal(6);
        });
      });

    });

    describe('addonLintTree', function() {
      beforeEach(function() {
        addon = { };

        project.initializeAddons = function() {
          this.addons = [ addon ];
        };

        emberApp = new EmberApp({
          project: project
        });
      });

      it('does not throw an error if lintTree is not defined', function() {
        emberApp.addonLintTree();
      });

      it('calls lintTree on the addon', function() {
        var actualType, actualTree;

        addon.lintTree = function(type, tree) {
          actualType = type;
          actualTree = tree;

          return 'blazorz';
        };

        var assertionsWereRun;

        mergeTreesStub = function(inputTree, options) {
          expect(inputTree).to.deep.equal(['blazorz']);
          expect(options).to.deep.equal({
            overwrite: true,
            annotation: 'TreeMerger (lint)'
          });

          assertionsWereRun = true;
        };

        emberApp.addonLintTree('blah', 'blam');

        expect(actualType).to.equal('blah');
        expect(actualTree).to.equal('blam');
        expect(assertionsWereRun).to.be.true;
      });

      it('filters out tree if lintTree returns falsey', function() {
        addon.lintTree = function() {
          return false;
        };

        var assertionsWereRun;

        mergeTreesStub = function(inputTree) {
          expect(inputTree.length).to.equal(0);

          assertionsWereRun = true;
        };

        emberApp.addonLintTree();

        expect(assertionsWereRun).to.be.true;
      });
    });
  });

  describe('import', function() {
    it('appends dependencies', function() {
      emberApp = new EmberApp({
      });
      emberApp.import('vendor/moment.js', {type: 'vendor'});
      expect(emberApp.legacyFilesToAppend.indexOf('vendor/moment.js')).to.equal(emberApp.legacyFilesToAppend.length - 1);
    });
    it('prepends dependencies', function() {
      emberApp = new EmberApp({
      });
      emberApp.import('vendor/es5-shim.js', {type: 'vendor', prepend: true});
      expect(emberApp.legacyFilesToAppend.indexOf('vendor/es5-shim.js')).to.equal(0);
    });
    it('defaults to development if production is not set', function() {
      process.env.EMBER_ENV = 'production';
      emberApp = new EmberApp({
      });
      emberApp.import({
        'development': 'vendor/jquery.js'
      });
      expect(emberApp.legacyFilesToAppend.indexOf('vendor/jquery.js')).to.equal(emberApp.legacyFilesToAppend.length -1);
      process.env.EMBER_ENV = undefined;

    });
    it('honors explicitly set to null in environment', function() {
      process.env.EMBER_ENV = 'production';
      emberApp = new EmberApp({
      });
      emberApp.import({
        'development': 'vendor/jquery.js',
        'production':  null
      });
      expect(emberApp.legacyFilesToAppend.indexOf('vendor/jquery.js')).to.equal(-1);
      process.env.EMBER_ENV = undefined;
    });
  });

  describe('vendorFiles', function() {
    var defaultVendorFiles = [
      'loader.js', 'jquery.js', 'ember.js',
      'app-shims.js', 'ember-resolver.js', 'ember-load-initializers.js'
    ];

    describe('handlebars.js', function() {
      it('does not app.import handlebars if not present in bower.json', function() {
        var app = new EmberApp({
          project: project
        });

        expect(app.vendorFiles).not.to.include.keys('handlebars.js');
      });

      it('includes handlebars if present in bower.json', function() {
        projectPath = path.resolve(__dirname, '../../fixtures/project-with-handlebars');
        project = setupProject(projectPath);

        var app = new EmberApp({
          project: project
        });

        expect(app.vendorFiles).to.include.keys('handlebars.js');
      });

      it('includes handlebars if present in provided `vendorFiles`', function() {
        var app = new EmberApp({
          project: project,
          vendorFiles: {
            'handlebars.js': 'some/path/whatever.js'
          }
        });

        expect(app.vendorFiles).to.include.keys('handlebars.js');
      });
    });

    it('defines vendorFiles by default', function() {
      emberApp = new EmberApp();
      expect(Object.keys(emberApp.vendorFiles)).to.deep.equal(defaultVendorFiles);
    });

    it('redefines a location of a vendor asset', function() {
      emberApp = new EmberApp({
        vendorFiles: {
          'ember.js': 'vendor/ember.js'
        }
      });
      expect(emberApp.vendorFiles['ember.js']).to.equal('vendor/ember.js');
    });

    it('defines vendorFiles in order even when option for it is passed', function() {
      emberApp = new EmberApp({
        vendorFiles: {
          'ember.js': 'vendor/ember.js'
        }
      });
      expect(Object.keys(emberApp.vendorFiles)).to.deep.equal(defaultVendorFiles);
    });

    it('removes dependency in vendorFiles', function() {
      emberApp = new EmberApp({
        vendorFiles: {
          'ember.js': null,
          'handlebars.js': null
        }
      });
      var vendorFiles = Object.keys(emberApp.vendorFiles);
      expect(vendorFiles.indexOf('ember.js')).to.equal(-1);
      expect(vendorFiles.indexOf('handlebars.js')).to.equal(-1);
    });

    it('defaults to ember.debug.js if exists in bower_components', function () {
      var root = path.resolve(__dirname, '../../fixtures/app/with-default-ember-debug');

      emberApp = new EmberApp({
        project: new Project(root, {})
      });

      var emberFiles = emberApp.vendorFiles['ember.js'];
      expect(emberFiles.development).to.equal('bower_components/ember/ember.debug.js');
    });

    it('switches the default ember.debug.js to ember.js if it does not exist', function () {
      emberApp = new EmberApp();
      var emberFiles = emberApp.vendorFiles['ember.js'];
      expect(emberFiles.development).to.equal('bower_components/ember/ember.js');
    });

    it('does not clobber an explicitly configured ember development file', function () {
      emberApp = new EmberApp({
        vendorFiles: {
          'ember.js': {
            development: 'vendor/ember.debug.js'
          }
        }
      });
      var emberFiles = emberApp.vendorFiles['ember.js'];
      expect(emberFiles.development).to.equal('vendor/ember.debug.js');
    });
  });
});
