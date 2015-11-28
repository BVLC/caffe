'use strict';

/**
@module ember-cli
*/

var existsSync   = require('exists-sync');
var path         = require('path');
var deprecate    = require('../utilities/deprecate');
var assign       = require('lodash/object/assign');
var SilentError  = require('silent-error');
var Reexporter   = require('../utilities/reexport');
var debug        = require('debug')('ember-cli:addon');

var nodeModulesPath = require('node-modules-path');

var p                   = require('ember-cli-preprocess-registry/preprocessors');
var preprocessJs        = p.preprocessJs;
var preprocessCss       = p.preprocessCss;
var preprocessTemplates = p.preprocessTemplates;

var AddonDiscovery  = require('../models/addon-discovery');
var AddonsFactory   = require('../models/addons-factory');
var CoreObject = require('core-object');
var Project = require('./project');

var upstreamMergeTrees = require('broccoli-merge-trees');
var Funnel     = require('broccoli-funnel');
var walkSync   = require('walk-sync');

var WatchedDir   = require('broccoli-source').WatchedDir;
var UnwatchedDir = require('broccoli-source').UnwatchedDir;

function mergeTrees(inputTree, options) {
  options = options || {};
  options.description = options.annotation;
  var tree = upstreamMergeTrees(inputTree, options);

  tree.description = options && options.description;

  return tree;
}


/**
  Root class for an Addon. If your addon module exports an Object this
  will be extended from this base class. If you export a constructor (function),
  it will **not** extend from this class.

  Hooks:

  - {{#crossLink "Addon/config:method"}}config{{/crossLink}}
  - {{#crossLink "Addon/blueprintsPath:method"}}blueprintsPath{{/crossLink}}
  - {{#crossLink "Addon/includedCommands:method"}}includedCommands{{/crossLink}}
  - {{#crossLink "Addon/serverMiddleware:method"}}serverMiddleware{{/crossLink}}
  - {{#crossLink "Addon/postBuild:method"}}postBuild{{/crossLink}}
  - {{#crossLink "Addon/outputReady:method"}}outputReady{{/crossLink}}
  - {{#crossLink "Addon/preBuild:method"}}preBuild{{/crossLink}}
  - {{#crossLink "Addon/buildError:method"}}buildError{{/crossLink}}
  - {{#crossLink "Addon/included:method"}}included{{/crossLink}}
  - {{#crossLink "Addon/postprocessTree:method"}}postprocessTree{{/crossLink}}
  - {{#crossLink "Addon/treeFor:method"}}treeFor{{/crossLink}}

  @class Addon
  @extends CoreObject
  @constructor
  @param {(Project|Addon)} parent The project or addon that directly depends on this addon
  @param {Project} project The current project (deprecated)
*/
function Addon(parent, project) {
  this.parent = parent;
  this.project = project;
  this.ui = project && project.ui;
  this.addonPackages = {};
  this.addons = [];
  this.addonDiscovery = new AddonDiscovery(this.ui);
  this.addonsFactory = new AddonsFactory(this, this.project);
  this.registry = p.defaultRegistry(this);
  this._didRequiredBuildPackages = false;

  if (!this.root) {
    throw new Error('Addon classes must be instantiated with the `root` property');
  }
  this.nodeModulesPath = nodeModulesPath(this.root);

  this.treePaths = {
    app:               'app',
    styles:            'app/styles',
    templates:         'app/templates',
    addon:             'addon',
    'addon-styles':    'addon/styles',
    'addon-templates': 'addon/templates',
    vendor:            'vendor',
    'test-support':    'test-support',
    public:            'public'
  };

  this.treeForMethods = {
    app:               'treeForApp',
    styles:            'treeForStyles',
    templates:         'treeForTemplates',
    'addon-templates': 'treeForAddonTemplates',
    addon:             'treeForAddon',
    vendor:            'treeForVendor',
    'test-support':    'treeForTestSupport',
    public:            'treeForPublic'
  };

  p.setupRegistry(this);

  if (!this.name) {
    throw new SilentError('An addon must define a `name` property.');
  }
}

Addon.__proto__ = CoreObject;
Addon.prototype.constructor = Addon;

Addon.prototype.hintingEnabled = function() {
  var isProduction = process.env.EMBER_ENV === 'production';
  var testsEnabledDefault = process.env.EMBER_CLI_TEST_COMMAND || !isProduction;

  return testsEnabledDefault;
};

/**
  Loads all required modules for a build

  @private
  @method _requireBuildPackages
 */

Addon.prototype._requireBuildPackages = function() {
  if (this._didRequiredBuildPackages === true) {
    return;
  } else {
    this._didRequiredBuildPackages = true;
  }

  this.transpileModules = deprecatedAddonFilters(this, 'this.transpileModules', 'broccoli-es6modules', function(tree, options) {
    return new (require('broccoli-es6modules'))(tree, options);
  });

  this.pickFiles = deprecatedAddonFilters(this, 'this.pickFiles', 'broccoli-funnel', function(tree, options) {
    return new Funnel(tree, options);
  });


  this.Funnel = deprecatedAddonFilters(this, 'new this.Funnel(..)', 'broccoli-funnel', function(tree, options) {
    return new Funnel(tree, options);
  });

  this.mergeTrees = deprecatedAddonFilters(this, 'this.mergeTrees', 'broccoli-merge-trees', mergeTrees);
  this.walkSync = deprecatedAddonFilters(this, 'this.walkSync', 'node-walk-sync', walkSync);
};

function deprecatedAddonFilters(addon, name, insteadUse, fn) {
  return function(tree, options) {
    var message = '[Deprecated] ' + name + ' is deprecated, please use ' + insteadUse + ' directly instead  [addon: ' + addon.name + ']';
    this.ui && this.ui.writeLine(message);

    if (!this.ui)  {
      console.warn(message);
    }

    return fn(tree, options);
  };
}

/**
  Shorthand method for [broccoli-sourcemap-concat](https://github.com/ef4/broccoli-sourcemap-concat)

  @private
  @method concatFiles
  @param {tree} tree Tree of files
  @param {Object} options Options for broccoli-sourcemap-concat
  @return {tree} Modified tree
*/
Addon.prototype.concatFiles = function(tree, options) {
  options.sourceMapConfig = this.app.options.sourcemaps;
  return require('broccoli-sourcemap-concat')(tree, options);
};

/**
  Returns whether or not this addon is running in development

  @public
  @method isDevelopingAddon
  @return {Boolean}
*/
Addon.prototype.isDevelopingAddon = function() {
  if (process.env.EMBER_ADDON_ENV === 'development' && (this.parent instanceof Project)) {
    return this.parent.name() === this.name;
  }
  return false;
};

/**
  Discovers all child addons of this addon and stores their names and
  package.json contents in this.addonPackages as key-value pairs

  @private
  @method discoverAddons
 */
Addon.prototype.discoverAddons = function() {
  var addonsList = this.addonDiscovery.discoverChildAddons(this);

  this.addonPackages = this.addonDiscovery.addonPackages(addonsList);
};

Addon.prototype.initializeAddons = function() {
  if (this._addonsInitialized) {
    return;
  }
  this._addonsInitialized = true;

  debug('initializeAddons for: %s', this.name);

  this.discoverAddons();
  this.addons = this.addonsFactory.initializeAddons(this.addonPackages);

  this.addons.forEach(function(addon) {
    debug('addon: %s', addon.name);
  });
};

/**
  Invoke the specified method for each enabled addon.

  @private
  @method eachAddonInvoke
  @param {String} methodName the method to invoke on each addon
  @param {Array} args the arguments to pass to the invoked method
*/
Addon.prototype.eachAddonInvoke = function eachAddonInvoke(methodName, args) {
  this.initializeAddons();

  var invokeArguments = args || [];

  return this.addons.map(function(addon) {
    if (addon[methodName]) {
      return addon[methodName].apply(addon, invokeArguments);
    }
  }).filter(Boolean);
};
/**
  Generates a tree for the specified path

  @private
  @method treeGenerator
  @return {tree}
*/
Addon.prototype.treeGenerator = function(dir) {
  var tree;
  if (this.project._watchmanInfo.canNestRoots || this.isDevelopingAddon()) {
    tree = new WatchedDir(dir);
  } else {
    tree = new UnwatchedDir(dir);
  }

  return tree;
};

/**
  Returns a given type of tree (if present), merged with the
  application tree. For each of the trees available using this
  method, you can also use a direct method called treeFor[Type] (eg. `treeForApp`).

  Available tree names:
  - app
  - styles
  - templates
  - {{#crossLink "Addon/treeForAddon:method"}}addon{{/crossLink}}
  - vendor
  - test-support
  - {{#crossLink "Addon/treeForPublic:method"}}public{{/crossLink}}

  @public
  @method treeFor
  @param {String} name
  @return {tree}
*/
Addon.prototype.treeFor = function treeFor(name) {
  this._requireBuildPackages();

  var tree;
  var trees = this.eachAddonInvoke('treeFor', [name]);

  if (tree = this._treeFor(name)) {
    trees.push(tree);
  }

  if (this.isDevelopingAddon() && this.hintingEnabled() && name === 'app') {
    trees.push(this.jshintAddonTree());
  }

  return mergeTrees(trees.filter(Boolean), {
    overwrite: true,
    annotation: 'Addon#treeFor (' + this.name + ' - ' + name + ')'
  });
};

/**
  @private
  @param {String} name
  @method _treeFor
  @return {tree}
*/
Addon.prototype._treeFor = function _treeFor(name) {
  var treePath = path.resolve(this.root, this.treePaths[name]);
  var treeForMethod = this.treeForMethods[name];
  var tree;

  if (existsSync(treePath)) {
    tree = this.treeGenerator(treePath);
  }

  if (this[treeForMethod]) {
    tree = this[treeForMethod](tree);
  }

  return tree;
};

/**
  This method is called when the addon is included in a build. You
  would typically use this hook to perform additional imports

  ```js
    included: function(app) {
      app.import(somePath);
    }
  ```

  @public
  @method included
  @param {EmberApp} app The application object
*/
Addon.prototype.included = function(/* app */) {
  if (!this._addonsInitialized) {
    // someone called `this._super.included` without `apply` (because of older
    // core-object issues that prevent a "real" super call from working properly)
    return;
  }

  this.eachAddonInvoke('included', [this]);
};

/**
  Returns the tree for all public files

  @public
  @method treeForPublic
  @param {Tree} tree
  @return {Tree} Public file tree
*/
Addon.prototype.treeForPublic = function(tree) {
  this._requireBuildPackages();

  if (!tree) {
    return tree;
  }

  return new Funnel(tree, {
    srcDir: '/',
    destDir: '/' + this.moduleName()
  });
};

/**
  Returns a tree for this addon

  @public
  @method treeForAddon
  @param {Tree} tree
  @return {Tree} Addon file tree
*/
Addon.prototype.treeForAddon = function(tree) {
  this._requireBuildPackages();

  if (!tree) {
    return tree;
  }

  var addonTree = this.compileAddon(tree);
  var stylesTree = this.compileStyles(this._treeFor('addon-styles'));

  return mergeTrees([addonTree, stylesTree].filter(Boolean), {
    annotation: 'Addon#treeForAddon(' + this.name + ')'
  });
};

/**
  @method treeForStyles
  @param {Tree} tree The tree to process, usually `app/styles/` in the addon.
  @returns {Tree} The return tree has the same contents as the input tree, but is moved so that the `app/styles/` path is preserved.
*/
Addon.prototype.treeForStyles = function(tree) {
  this._requireBuildPackages();

  if (!tree) {
    return tree;
  }

  return new Funnel(tree, {
    destDir: 'app/styles'
  });
};

/**
  Runs the styles tree through preprocessors.

  @private
  @method compileStyles
  @param {Tree} tree Styles file tree
  @return {Tree} Compiled styles tree
*/
Addon.prototype.compileStyles = function(tree) {
  this._requireBuildPackages();

  if (tree) {
    return preprocessCss(tree, '/', '/', {
      outputPaths: { 'addon': this.name + '.css' },
      registry: this.registry
    });
  }
};

/**
  Looks in the addon/ and addon/templates trees to determine if template files
  exists that need to be precompiled.

  This is executed once when building, but not on rebuilds.

  @private
  @method shouldCompileTemplates
  @returns Boolean indicates if templates need to be compiled for this addon
*/
Addon.prototype.shouldCompileTemplates = function() {
  var templateExtensions = this.registry.extensionsForType('template');
  var addonTreePath = path.join(this.root, this.treePaths['addon']);
  var addonTemplatesTreePath = path.join(this.root, this.treePaths['addon-templates']);

  var files = [];

  if (existsSync(addonTreePath)) {
    files = files.concat(walkSync(addonTreePath));
  }

  if (existsSync(addonTemplatesTreePath)) {
    files = files.concat(walkSync(addonTemplatesTreePath));
  }

  var extensionMatcher = new RegExp('(' + templateExtensions.join('|') + ')$');

  return files.some(function(file) {
    return file.match(extensionMatcher);
  });
};

/**
  Runs the templates tree through preprocessors.

  @private
  @method compileTemplates
  @param {Tree} tree Templates file tree
  @return {Tree} Compiled templates tree
*/
Addon.prototype.compileTemplates = function(tree) {
  this._requireBuildPackages();

  if (this.shouldCompileTemplates()) {
    var plugins = this.registry.load('template');

    if (plugins.length === 0) {
      throw new SilentError('Addon templates were detected, but there ' +
                            'are no template compilers registered for `' + this.name + '`. ' +
                            'Please make sure your template precompiler (commonly `ember-cli-htmlbars`) ' +
                            'is listed in `dependencies` (NOT `devDependencies`) in ' +
                            '`' + this.name + '`\'s `package.json`.');
    }

    var addonTemplates = this._treeFor('addon-templates');
    var standardTemplates;

    if (addonTemplates) {
      standardTemplates = new Funnel(addonTemplates, {
        srcDir: '/',
        destDir: 'modules/' + this.name + '/templates'
      });
    }

    var includePatterns = this.registry.extensionsForType('template').map(function(extension) {
      return '**/*/template.' + extension;
    });

    var podTemplates = new Funnel(tree, {
      include: includePatterns,
      destDir: 'modules/' + this.name + '/',
      annotation: 'Funnel: Addon Pod Templates'
    });

    return preprocessTemplates(mergeTrees([standardTemplates, podTemplates].filter(Boolean), {
      annotation: 'compileTemplates(' + this.name + ')'
    }), {
      registry: this.registry
    });
  }
};

/**
  Runs the addon tree through preprocessors.

  @private
  @method compileAddon
  @param {Tree} tree Addon file tree
  @return {Tree} Compiled addon tree
*/
Addon.prototype.compileAddon = function(tree) {
  this._requireBuildPackages();

  var addonJs = this.processedAddonJsFiles(tree);
  var templatesTree = this.compileTemplates(tree);
  var reexported = new Reexporter(this.name, this.name + '.js');
  var trees = [addonJs, templatesTree, reexported].filter(Boolean);

  return mergeTrees(trees, {
    annotation: 'Addon#compileAddon(' + this.name + ') '
  });
};

/**
  Returns a tree with JSHhint output for all addon JS.

  @private
  @method jshintAddonTree
  @return {Tree} Tree with JShint output (tests)
*/
Addon.prototype.jshintAddonTree = function() {
  this._requireBuildPackages();

  var addonPath = path.join(this.root, this.treePaths['addon']);

  if (!existsSync(addonPath)) {
    return;
  }

  var addonJs = this.addonJsFiles(addonPath);
  var jshintedAddon = this.app.addonLintTree('addon', addonJs);

  return new Funnel(jshintedAddon, {
    srcDir: '/',
    destDir: this.name + '/tests/'
  });
};

/**
  Returns a tree containing the addon's js files

  @private
  @method addonJsFiles
  @return {Tree} The filtered addon js files
*/
Addon.prototype.addonJsFiles = function(tree) {
  this._requireBuildPackages();

  var includePatterns = this.registry.extensionsForType('js').map(function(extension) {
    return new RegExp(extension + '$');
  });

  return new Funnel(tree, {
    include: includePatterns,
    destDir: 'modules/' + this.moduleName(),
    description: 'Funnel: Addon JS'
  });
},


/**
  Preprocesses a javascript tree.

  @private
  @method preprocessJs
  @return {Tree} Preprocessed javascript
*/
Addon.prototype.preprocessJs = function() {
  return preprocessJs.apply(preprocessJs, arguments);
},

/**
  Returns a tree with all javascript for this addon.

  @private
  @method processedAddonJsFiles
  @param {Tree} the tree to preprocess
  @return {Tree} Processed javascript file tree
*/
Addon.prototype.processedAddonJsFiles = function(tree) {
  return this.preprocessJs(this.addonJsFiles(tree), '/', this.name, {
    registry: this.registry
  });
};

/**
  Returns the module name for this addon.

  @public
  @method moduleName
  @return {String} module name
*/
Addon.prototype.moduleName = function() {
  if (!this.modulePrefix) {
    this.modulePrefix = (this.modulePrefix || this.name).toLowerCase().replace(/\s/g, '-');
  }

  return this.modulePrefix;
};

/**
  Returns the path for addon blueprints.

  @private
  @method blueprintsPath
  @return {String} The path for blueprints
*/
Addon.prototype.blueprintsPath = function() {
  var blueprintPath = path.join(this.root, 'blueprints');

  if (existsSync(blueprintPath)) {
    return blueprintPath;
  }
};

/**
  Augments the applications configuration settings.
  Object returned from this hook is merged with the application's configuration object.
  Application's configuration always take precedence.


  ```js
    config: function(environment, appConfig) {
      return {
        someAddonDefault: "foo"
      };
    }
  ```

  @public
  @method config
  @param {String} env Name of current environment (ie "developement")
  @param {Object} baseConfig Initial application configuration
  @return {Object} Configuration object to be merged with application configuration.
*/
Addon.prototype.config = function (env, baseConfig) {
  var configPath = path.join(this.root, 'config', 'environment.js');

  if (existsSync(configPath)) {
    var configGenerator = require(configPath);

    return configGenerator(env, baseConfig);
  }
};

/**
  @public
  @method dependencies
  @return {Object} The addon's dependencies based on the addon's package.json
*/
Addon.prototype.dependencies = function() {
  var pkg = this.pkg || {};
  return assign({}, pkg['devDependencies'], pkg['dependencies']);
};

/**
  @public
  @method isEnabled
  @return {Boolean} Whether or not this addon is enabled
*/
Addon.prototype.isEnabled = function() {
  return true;
};

/**
  Returns the absolute path for a given addon

  @private
  @method resolvePath
  @param {String} addon Addon name
  @return {String} Absolute addon path
*/
Addon.resolvePath = function(addon) {
  var addonMain;

  deprecate(addon.pkg.name + ' is using the deprecated ember-addon-main definition. It should be updated to {\'ember-addon\': {\'main\': \'' + addon.pkg['ember-addon-main'] + '\'}}', addon.pkg['ember-addon-main']);

  addonMain = addon.pkg['ember-addon-main'] || (addon.pkg['ember-addon'] && addon.pkg['ember-addon'].main) || addon.pkg['main'] || 'index.js';

  // Resolve will fail unless it has an extension
  if(!path.extname(addonMain)) {
    addonMain += '.js';
  }

  return path.resolve(addon.path, addonMain);
};

/**
  Returns the addon class for a given addon name.
  If the Addon exports a function, that function is used
  as constructor. If an Object is exported, a subclass of
  `Addon` is returned with the exported hash merged into it.

  @private
  @static
  @method lookup
  @param {String} addon Addon name
  @return {Addon} Addon class
*/
Addon.lookup = function(addon) {
  var Constructor, addonModule, modulePath, moduleDir;

  modulePath = Addon.resolvePath(addon);
  moduleDir  = path.dirname(modulePath);

  if (existsSync(modulePath)) {
    addonModule = require(modulePath);

    if (typeof addonModule === 'function') {
      Constructor = addonModule;
      Constructor.prototype.root = Constructor.prototype.root || moduleDir;
      Constructor.prototype.pkg  = Constructor.prototype.pkg || addon.pkg;
    } else {
      Constructor = Addon.extend(assign({
        root: moduleDir,
        pkg: addon.pkg
      }, addonModule));
    }
  }

  if (!Constructor) {
    throw new SilentError('The `' + addon.pkg.name + '` addon could not be found at `' + addon.path + '`.');
  }

  return Constructor;
};


// Methods without default implementation

/**
  CLI commands included with this addon. This function should return
  an object with command names and command instances/create options.

  This function is not implemented by default

  ```js
    includedCommands: function() {
      return {
        'do-foo': require('./lib/commands/foo')
      };
    }
  ```

  @public
  @method includedCommands
  @return {Object} An object with included commands
*/


/**
  Post-process a tree

  @public
  @method postProcessTree
  @param {String} type What kind of tree (eg. 'javascript', 'styles')
  @param {Tree} Tree to process
  @return {Tree} Processed tree
*/

/**
  This hook allows you to make changes to the express server run by ember-cli.
  It's passed an `startOptions` object which contains:
  - `app` Express server instance
  - `options` A hash with:
    - `project` Current {{#crossLink "Project"}}project{{/crossLink}}
    - `watcher`
    - `environment`

  This function is not implemented by default

  ```js
    serverMiddleware: function(startOptions) {
      var app = startOptions.app;

      app.use(function(req, res, next) {
        // Some middleware
      });
    }
  ```

  @public
  @method serverMiddleware
  @param {Object} startOptions Express server start options
*/

/**
  This hook is called before a build takes place.

  @public
  @method preBuild
  @param {Object} result Build object
*/

/**
  This hook is called after a build is complete.

  It's passed an `result` object which contains:
  - `directory` Path to build output

  @public
  @method postBuild
  @param {Object} result Build result object
*/

/**
  This hook is called after the build files have been copied to the output directory

  It's passed an `result` object which contains:
  - `directory` Path to build output

  @public
  @method outputReady
  @param {Object} result Build result object
*/

/**
  This hook is called when an error occurs during the preBuild, postBuild or outputReady hooks
  for addons, or when the build fails

  @public
  @method buildError
  @param {Error} error The error that was caught during the processes listed above
*/

module.exports = Addon;
