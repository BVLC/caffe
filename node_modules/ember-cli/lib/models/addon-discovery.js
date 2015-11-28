'use strict';

/**
@module ember-cli
*/

var assign     = require('lodash/object/assign');
var debug      = require('debug')('ember-cli:addon-discovery');
var existsSync = require('exists-sync');
var path       = require('path');
var CoreObject = require('core-object');
var resolve    = require('resolve');
var findup     = require('findup');

/**
  AddonDiscovery is responsible for collecting information about all of the
  addons that will be used with a project.

  @class AddonDiscovery
  @extends CoreObject
  @constructor
*/

function AddonDiscovery(ui) {
  this.ui = ui;
}

AddonDiscovery.__proto__ = CoreObject;
AddonDiscovery.prototype.constructor = AddonDiscovery;

/**
  This is one of the primary APIs for this class and is called by the project.
  It returns a tree of plain objects that each contain information about a
  discovered addon. Each node has `name`, `path`, `pkg` and
  `childAddons` properties. The latter is an array containing any addons
  discovered from applying the discovery process to that addon.

  @private
  @method discoverProjectAddons
 */
AddonDiscovery.prototype.discoverProjectAddons = function(project) {
  var projectAsAddon = this.discoverFromProjectItself(project);
  var internalAddons = this.discoverFromInternalProjectAddons(project);
  var cliAddons = this.discoverFromCli(project.cli);
  var dependencyAddons;

  if (project.hasDependencies()) {
    dependencyAddons = this.discoverFromDependencies(project.root, project.nodeModulesPath, project.pkg, false);
  }  else {
    dependencyAddons = [];
  }

  var inRepoAddons = this.discoverInRepoAddons(project.root, project.pkg);
  var addons = projectAsAddon.concat(cliAddons, internalAddons, dependencyAddons, inRepoAddons);

  return addons;
};

/**
  This is one of the primary APIs for this class and is called by addons.
  It returns a tree of plain objects that each contain information about a
  discovered addon. Each node has `name`, `path`, `pkg` and
  `childAddons` properties. The latter is an array containing any addons
  discovered from applying the discovery process to that addon.

  @private
  @method discoverProjectAddons
 */
AddonDiscovery.prototype.discoverChildAddons = function(addon) {
  debug('discoverChildAddons: %s(%s)', addon.name, addon.root);
  var dependencyAddons = this.discoverFromDependencies(addon.root, addon.nodeModulesPath, addon.pkg, true);
  var inRepoAddons = this.discoverInRepoAddons(addon.root, addon.pkg);
  var addons = dependencyAddons.concat(inRepoAddons);
  return addons;
};

/**
  Returns an array containing zero or one nodes, depending on whether or not
  the passed project is an addon.

  @private
  @method discoverFromProjectItself
 */
AddonDiscovery.prototype.discoverFromProjectItself = function(project) {
  if (project.isEmberCLIAddon()) {
    var addonPkg = this.discoverAtPath(project.root);
    if (addonPkg) {
      return [addonPkg];
    }
  }
  return [];
};

/**
  Returns a tree based on the addons referenced in the provided `pkg` through
  the package.json `dependencies` and optionally `devDependencies` collections,
  as well as those discovered addons' child addons.

  @private
  @method discoverFromDependencies
 */
AddonDiscovery.prototype.discoverFromDependencies = function(root, nodeModulesPath, pkg, excludeDevDeps) {
  var discovery = this;
  var addons = Object.keys(this.dependencies(pkg, excludeDevDeps)).map(function(name) {
    if (name !== 'ember-cli') {
      var addonPath = this.resolvePackage(root, name);

      if (addonPath) {
        return discovery.discoverAtPath(addonPath);
      }

      // this supports packages that do not have a valid entry point
      // script (aka `main` entry in `package.json` or `index.js`)
      addonPath = path.join(nodeModulesPath, name);
      var addon = discovery.discoverAtPath(addonPath);
      if (addon) {
        var chalk = require('chalk');

        discovery.ui.writeLine(chalk.yellow('The package `' + name + '` is not a properly formatted package, we have used a fallback lookup to resolve it at `' + addonPath + '`. This is generally caused by an addon not having a `main` entry point (or `index.js`).'), 'WARNING');

        return addon;
      }
    }
  }, this).filter(Boolean);
  return addons;
};

AddonDiscovery.prototype.resolvePackage = function(root, packageName) {
  try {

    var entryModulePath = resolve.sync(packageName, { basedir: root });

    return findup.sync(entryModulePath, 'package.json');
  } catch(e) {
    var acceptableError = 'Cannot find module \'' + packageName + '\' from \'' + root + '\'';
    // pending: https://github.com/substack/node-resolve/pull/80
    var workAroundError = 'Cannot read property \'isFile\' of undefined';

    if (e.message === workAroundError || e.message === acceptableError) {
      return;
    }
    throw e;
  }
};

/**
  Returns a tree based on the in-repo addons referenced in the provided `pkg`
  through paths listed in the `ember-addon` entry, as well as those discovered
  addons' child addons.

  @private
  @method discoverInRepoAddons
 */
AddonDiscovery.prototype.discoverInRepoAddons = function(root, pkg) {
  if (!pkg || !pkg['ember-addon'] || !pkg['ember-addon'].paths) {
    return [];
  }
  var discovery = this;
  var addons = pkg['ember-addon'].paths.map(function(addonPath) {
    addonPath = path.join(root, addonPath);
    return discovery.discoverAtPath(addonPath);
  }, this).filter(Boolean);
  return addons;
};

/**
  Returns a tree based on the internal addons that may be defined within the project.
  It does this by consulting the projects `supportedInternalAddonPaths()` method, which
  is primarily used for middleware addons.

  @private
  @method discoverFromInternalProjectAddons
 */
AddonDiscovery.prototype.discoverFromInternalProjectAddons = function(project) {
  var discovery = this;
  return project.supportedInternalAddonPaths().map(function(path){
    return discovery.discoverAtPath(path);
  }).filter(Boolean);
};

AddonDiscovery.prototype.discoverFromCli = function (cli) {
  if (!cli) { return []; }

  var cliPkg = require(path.resolve(cli.root, 'package.json'));
  return this.discoverInRepoAddons(cli.root, cliPkg);
};

/**
  Given a particular path, return undefined if the path is not an addon, or if it is,
  a node with the info about the addon.

  @private
  @method discoverAtPath
 */
AddonDiscovery.prototype.discoverAtPath = function(addonPath) {
  var pkgPath = path.join(addonPath, 'package.json');
  debug('attemping to add: %s',  addonPath);

  if (existsSync(pkgPath)) {
    var addonPkg = require(pkgPath);
    var keywords = addonPkg.keywords || [];
    debug(' - module found: %s', addonPkg.name);

    addonPkg['ember-addon'] = addonPkg['ember-addon'] || {};

    if (keywords.indexOf('ember-addon') > -1) {
      debug(' - is addon, adding...');
      var addonInfo = {
        name: addonPkg.name,
        path: addonPath,
        pkg: addonPkg,
      };
      return addonInfo;
    } else {
      debug(' - no ember-addon keyword, not including.');
    }
  } else {
    debug(' - no package.json (looked at ' + pkgPath + ').');
  }

  return null;
};

/**
  Returns the dependencies from a package.json

  @private
  @method dependencies
  @param  {Object}  pkg            Package object. If false, the current package is used.
  @param  {Boolean} excludeDevDeps Whether or not development dependencies should be excluded, defaults to false.
  @return {Object}                 Dependencies
 */
AddonDiscovery.prototype.dependencies = function(pkg, excludeDevDeps) {
  pkg = pkg || {};

  var devDependencies = pkg['devDependencies'];
  if (excludeDevDeps) {
    devDependencies = {};
  }

  return assign({}, devDependencies, pkg['dependencies']);
};

AddonDiscovery.prototype.addonPackages = function(addonsList) {
  var addonPackages = {};

  addonsList.forEach(function(addonPkg) {
    addonPackages[addonPkg.name] = addonPkg;
  });

  return addonPackages;
};

// Export
module.exports = AddonDiscovery;
