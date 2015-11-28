'use strict';

var path = require('path');
var expect = require('chai').expect;
var assign = require('lodash/object/assign');
var Project = require('../../../lib/models/project');
var AddonDiscovery = require('../../../lib/models/addon-discovery');
var fixturePath = path.resolve(__dirname, '../../fixtures/addon');
var MockUI = require('../../helpers/mock-ui');
var chalk = require('chalk');

describe('models/addon-discovery.js', function() {
  var project, projectPath, ui;
  this.timeout(40000);

  beforeEach(function() {
    ui = new MockUI();
    projectPath = path.resolve(fixturePath, 'simple');
    var packageContents = require(path.join(projectPath, 'package.json'));

    project = new Project(projectPath, packageContents, ui);
  });

  describe('dependencies', function() {
    var mockPkg, deps, devDeps;

    beforeEach(function() {
      deps = {
        'foo-bar': 'latest',
        'blah-blah': '1.0.0'
      };

      devDeps = {
        'dev-foo-bar': 'latest'
      };

      mockPkg = {
        dependencies: deps,
        devDependencies: devDeps
      };
    });

    it('returns an object containing depenencies from the provided package.json', function() {
      var expected = assign({}, deps, devDeps);
      var discovery = new AddonDiscovery(ui);

      expect(discovery.dependencies(mockPkg)).to.be.eql(expected);
    });

    it('excludes development dependencies if instructed', function() {
      var expected = assign({}, deps);
      var discovery = new AddonDiscovery(ui);

      expect(discovery.dependencies(mockPkg, true)).to.be.eql(expected);
    });
  });

  describe('discoverFromInternalProjectAddons', function() {
    it('calls discoverAtPath for each path in project.supportedInternalAddonPaths', function() {
      var actualPaths = [];
      var project = {
        supportedInternalAddonPaths: function() {
          return [ 'lib/foo/', 'baz/qux/' ];
        }
      };

      var discovery = new AddonDiscovery(ui);

      discovery.discoverAtPath = function(path) {
        actualPaths.push(path);
      };

      discovery.discoverFromInternalProjectAddons(project);

      expect(actualPaths).to.be.eql(project.supportedInternalAddonPaths());
    });
  });

  describe('discoverInRepoAddons', function() {
    describe('returns empty array when ember-addon.paths is empty', function() {
      var discovery, pkg;

      beforeEach(function() {
        discovery = new AddonDiscovery(ui);
      });

      it('returns empty array if `ember-addon` is not present in provided package', function() {
        pkg = { };

        var actual = discovery.discoverInRepoAddons(fixturePath, pkg);
        expect(actual).to.be.eql([]);
      });

      it('returns empty array if `ember-addon.paths` is missing in provided package', function() {
        pkg = {
          'ember-addon': { }
        };

        var actual = discovery.discoverInRepoAddons(fixturePath, pkg);
        expect(actual).to.be.eql([]);
      });

      it('returns empty array if `ember-addon.paths` is empty in provided package', function() {
        pkg = {
          'ember-addon': {
            paths: []
          }
        };

        var actual = discovery.discoverInRepoAddons(fixturePath, pkg);
        expect(actual).to.be.eql([]);
      });
    });

    it('calls discoverAtPath for each path in ember-addon.paths', function() {
      var actualPaths = [];
      var pkg = {
        'ember-addon': {
          paths: [ 'lib/foo', 'baz/qux' ]
        }
      };
      var discovery = new AddonDiscovery(ui);

      discovery.discoverAtPath = function(providedPath) {
        actualPaths.push(providedPath);

        return providedPath;
      };

      discovery.discoverInRepoAddons(fixturePath, pkg);

      var expected = [
        path.join(fixturePath, 'lib', 'foo'),
        path.join(fixturePath, 'baz', 'qux')
      ];

      expect(actualPaths).to.be.eql(expected);
    });

    it('falsey results from discoverAtPath are filtered out', function() {
      var actualPaths = [];
      var pkg = {
        'ember-addon': {
          paths: [ 'lib/foo', 'baz/qux' ]
        }
      };
      var discovery = new AddonDiscovery(ui);

      discovery.discoverAtPath = function(providedPath) {
        actualPaths.push(providedPath);

        return null;
      };

      var result = discovery.discoverInRepoAddons(fixturePath, pkg);

      var expectedPaths = [
        path.join(fixturePath, 'lib', 'foo'),
        path.join(fixturePath, 'baz', 'qux')
      ];

      expect(actualPaths).to.be.eql(expectedPaths);
      expect(result).to.be.eql([]);
    });
  });

  describe('discoverFromDependencies', function() {
    var mockPkg, deps, devDeps;

    beforeEach(function() {
      deps = {
        'foo-bar': 'latest',
        'blah-blah': '1.0.0'
      };

      devDeps = {
        'dev-foo-bar': 'latest'
      };

      mockPkg = {
        dependencies: deps,
        devDependencies: devDeps
      };
    });

    it('can find a package without a main entry point [DEPRECATED]', function() {
      var root = path.join(fixturePath, 'shared-package', 'base');
      var addonNodeModulesPath = path.join(root, 'node_modules');
      var actualPaths = [];
      var discovery = new AddonDiscovery(ui);

      deps['invalid-package'] = 'latest';
      discovery.discoverAtPath = function(providedPath) {
        actualPaths.push(providedPath);

        return providedPath;
      };

      discovery.discoverFromDependencies(root, addonNodeModulesPath, mockPkg, true);

      var expectedPaths = [
        path.join(root, 'node_modules', 'foo-bar'),
        path.join(root, 'node_modules', 'blah-blah'),
        path.join(root, 'node_modules', 'invalid-package')
      ];

      expect(actualPaths).to.be.eql(expectedPaths);

      var output = ui.output.trim();
      var expectedWarning = chalk.yellow('The package `invalid-package` is not a properly formatted package, we have used a fallback lookup to resolve it at `' + path.join(root, 'node_modules', 'invalid-package') + '`. This is generally caused by an addon not having a `main` entry point (or `index.js`).');
      expect(output).to.equal(expectedWarning);
    });

    it('does not error when dependencies are not found', function() {
      var root = path.join(fixturePath, 'shared-package', 'base');
      var addonNodeModulesPath = path.join(root, 'node_modules');
      var actualPaths = [];
      var discovery = new AddonDiscovery(ui);

      deps['blah-zorz'] = 'latest';
      discovery.discoverAtPath = function(providedPath) {
        actualPaths.push(providedPath);

        return providedPath;
      };

      discovery.discoverFromDependencies(root, addonNodeModulesPath, mockPkg, true);

      var expectedPaths = [
        path.join(root, 'node_modules', 'foo-bar'),
        path.join(root, 'node_modules', 'blah-blah'),
        path.join(root, 'node_modules', 'blah-zorz')
      ];

      expect(actualPaths).to.be.eql(expectedPaths);
    });

    it('calls discoverAtPath for each entry in dependencies', function() {
      var root = path.join(fixturePath, 'shared-package', 'base');
      var addonNodeModulesPath = path.join(root, 'node_modules');
      var actualPaths = [];
      var discovery = new AddonDiscovery(ui);

      discovery.discoverAtPath = function(providedPath) {
        actualPaths.push(providedPath);

        return providedPath;
      };

      discovery.discoverFromDependencies(root, addonNodeModulesPath, mockPkg);

      var expectedPaths = [
        path.join(root, '..', 'node_modules', 'dev-foo-bar'),
        path.join(root, 'node_modules', 'foo-bar'),
        path.join(root, 'node_modules', 'blah-blah')
      ];

      expect(actualPaths).to.be.eql(expectedPaths);
    });

    it('excludes devDeps if `excludeDevDeps` is true', function() {
      var root = path.join(fixturePath, 'shared-package', 'base');
      var addonNodeModulesPath = path.join(root, 'node_modules');
      var actualPaths = [];
      var discovery = new AddonDiscovery(ui);

      discovery.discoverAtPath = function(providedPath) {
        actualPaths.push(providedPath);

        return providedPath;
      };

      discovery.discoverFromDependencies(root, addonNodeModulesPath, mockPkg, true);

      var expectedPaths = [
        path.join(root, 'node_modules', 'foo-bar'),
        path.join(root, 'node_modules', 'blah-blah')
      ];

      expect(actualPaths).to.be.eql(expectedPaths);
    });
  });

  describe('discoverFromProjectItself', function() {
    it('adds the project.root if it is an addon', function() {
      var project = {
        isEmberCLIAddon: function() {
          return false;
        }
      };

      var discovery = new AddonDiscovery(ui);
      var actual = discovery.discoverFromProjectItself(project);

      expect(actual).to.be.eql([]);
    });

    it('returns the root path if the project is an addon', function() {
      var actualPaths = [];
      var project = {
        root: 'foo/bar/baz',
        isEmberCLIAddon: function() {
          return true;
        }
      };

      var discovery = new AddonDiscovery(ui);

      discovery.discoverAtPath = function(providedPath) {
        actualPaths.push(providedPath);

        return providedPath;
      };

      var actual = discovery.discoverFromProjectItself(project);
      var expectedPaths = [ 'foo/bar/baz' ];

      expect(actualPaths).to.be.eql(expectedPaths);
      expect(actual).to.be.eql(expectedPaths);
    });
  });

  describe('discoverChildAddons', function() {
    var addon, discovery, discoverFromDependenciesCalled, discoverInRepoAddonsCalled;

    beforeEach(function() {
      addon = {
        name: 'awesome-sauce',
        root: fixturePath,
        pkg: {
          dependencies: {
            'foo-bar': 'latest'
          },
          devDependencies: {
            'dev-dep-bar': 'latest'
          }
        }
      };

      discovery = new AddonDiscovery(ui);

      discovery.discoverFromDependencies = function() {
        discoverFromDependenciesCalled = true;

        return [];
      };

      discovery.discoverInRepoAddons = function() {
        discoverInRepoAddonsCalled = true;

        return [];
      };
    });

    it('delegates to discoverInRepoAddons and discoverFromDependencies', function() {
      discovery.discoverChildAddons(addon);

      expect(discoverInRepoAddonsCalled).to.equal(true);
      expect(discoverFromDependenciesCalled).to.equal(true);
    });

    it('concats  discoverInRepoAddons and discoverFromDependencies results', function() {
      discovery.discoverFromDependencies = function() {
        return [ 'discoverFromDependencies' ];
      };

      discovery.discoverInRepoAddons = function() {
        return [ 'discoverInRepoAddons' ];
      };

      var result = discovery.discoverChildAddons(addon);

      expect(result).to.be.eql([ 'discoverFromDependencies', 'discoverInRepoAddons' ]);
    });
  });

  describe('discoverProjectAddons', function() {
    var addon, discovery, discoverFromProjectItselfCalled, discoverFromInternalProjectAddonsCalled, discoverFromDependenciesCalled, discoverInRepoAddonsCalled;

    beforeEach(function() {
      addon = {
        name: 'awesome-sauce',
        root: fixturePath,
        pkg: {
          dependencies: {
            'foo-bar': 'latest'
          },
          devDependencies: {
            'dev-dep-bar': 'latest'
          }
        },
        hasDependencies: function() {
          return true;
        }
      };

      discovery = new AddonDiscovery(ui);

      discovery.discoverFromProjectItself = function() {
        discoverFromProjectItselfCalled = true;

        return [ 'discoverFromProjectItself' ];
      };

      discovery.discoverFromInternalProjectAddons = function() {
        discoverFromInternalProjectAddonsCalled = true;

        return [ 'discoverFromInternalProjectAddons' ];
      };

      discovery.discoverFromDependencies = function() {
        discoverFromDependenciesCalled = true;

        return [ 'discoverFromDependencies' ];
      };

      discovery.discoverInRepoAddons = function() {
        discoverInRepoAddonsCalled = true;

        return [ 'discoverInRepoAddons' ];
      };
    });

    it('delegates to internal methods', function() {
      discovery.discoverProjectAddons(addon);

      expect(discoverFromProjectItselfCalled).to.equal(true);
      expect(discoverFromInternalProjectAddonsCalled).to.equal(true);
      expect(discoverInRepoAddonsCalled).to.equal(true);
      expect(discoverFromDependenciesCalled).to.equal(true);
    });

    it('concats  discoverInRepoAddons and discoverFromDependencies results', function() {
      var result = discovery.discoverProjectAddons(addon);

      expect(result).to.be.eql([ 'discoverFromProjectItself', 'discoverFromInternalProjectAddons', 'discoverFromDependencies', 'discoverInRepoAddons' ]);
    });
  });

  describe('discoverAtPath', function() {
    it('returns an info object when addon is found', function() {
      var addonPath = path.join(fixturePath, 'simple/node_modules/ember-random-addon');
      var addonPkg = require(path.join(addonPath, 'package.json'));
      var discovery = new AddonDiscovery(ui);

      var result = discovery.discoverAtPath(addonPath);

      expect(result.name).to.be.equal('ember-random-addon');
      expect(result.path).to.be.equal(addonPath);
      expect(result.pkg).to.be.eql(addonPkg);
    });

    it('returns `null` if path is not for an addon', function() {
      var addonPath = path.join(fixturePath, 'simple');
      var discovery = new AddonDiscovery(ui);

      var result = discovery.discoverAtPath(addonPath);

      expect(result).to.be.equal(null);
    });
  });
});
