'use strict';

var path     = require('path');
var Registry = require('./');
var relativeRequire = require('process-relative-require');
var debug = require('debug')('ember-cli:preprocessors');

/**
  Invokes the `setupRegistryForEachAddon('parent', registry)` hook for each of the parent objects addons.

  @private
  @method setupRegistryForEachAddon
  @param {Registry} registry the registry being setup
  @param {Addon|EmberApp} parent the parent object of the registry being setup. Will be an addon for nested
    addons, or the `EmberApp` for addons in the project directly.
*/
function setupRegistryForEachAddon(registry, parent) {
  parent.initializeAddons();
  var addons = parent.addons || (parent.project && parent.project.addons);

  if (!addons) {
    return;
  }

  addons.forEach(function(addon) {
    if (addon.setupPreprocessorRegistry) {
      addon.setupPreprocessorRegistry('parent', registry);
    }
  });
}

/**
  Invokes the `setupPreprocessorRegistry` hook for a given addon. The `setupPreprocessorRegistry` will be
  invoked first on the addon itself (with the first argument of `'self'`), and then on each nested addon
  (with the first argument of `'parent'`).

  @private
  @method setupRegistry
  @param {Addon|EmberApp}
*/
module.exports.setupRegistry = function(appOrAddon) {
  var registry = appOrAddon.registry;
  if (appOrAddon.setupPreprocessorRegistry) {
    appOrAddon.setupPreprocessorRegistry('self', registry);
  }
  setupRegistryForEachAddon(registry, appOrAddon);

  addLegacyPreprocessors(registry);
};

/**
  Creates a Registry instance, and prepopulates it with a few static default
  preprocessors.

  @private
  @method defaultRegistry
  @param app
*/
module.exports.defaultRegistry = function(app) {
  var registry = new Registry(app.dependencies(), app);

  return registry;
};

/**
  Add old / grandfathered preprocessor that is not an ember-cli addon.

  These entries should be removed, once they have good addon replacements.
  @private
  @method addLegacyPreprocessors
  @param registry
*/
function addLegacyPreprocessors(registry) {
  registry.add('css', 'broccoli-stylus-single', 'styl');
  registry.add('css', 'broccoli-ruby-sass', ['scss', 'sass']);
  registry.add('css', 'broccoli-sass', ['scss', 'sass']);

  registry.add('minify-css', 'broccoli-csso', null);

  registry.add('js', 'broccoli-ember-script', 'em');

  registry.add('template', 'broccoli-emblem-compiler', ['embl', 'emblem']);
  registry.add('template', 'broccoli-ember-hbs-template-compiler', ['hbs', 'handlebars']);
}

/**
  Returns true if the given path would be considered of a specific type.

  For example:

  ```
  isType('somefile.js', 'js', addon); // => true
  isType('somefile.css', 'css', addon); // => true
  isType('somefile.blah', 'css', addon); // => false
  isType('somefile.sass', 'css', addon); // => true if a sass preprocessor is available
  ```
  @private
  @method isType
  @param {String} file the path to check
  @param {String} type the type to compare with
  @param {registryOwner} registryOwner the object whose registry we should search
*/
module.exports.isType = function(file, type, registryOwner) {
  var extension = path.extname(file).replace('.', '');

  if (extension === type) { return true; }

  if (registryOwner.registry.extensionsForType(type).indexOf(extension) > -1) {
    return true;
  }
};

module.exports.preprocessMinifyCss = function(tree, options) {
  var plugins = options.registry.load('minify-css');

  if (plugins.length === 0) {
    var compiler = require('broccoli-clean-css');
    return compiler(tree, options);
  } else if (plugins.length > 1) {
    throw new Error('You cannot use more than one minify-css plugin at once.');
  }

  var plugin = plugins[0];

  return relativeRequire(plugin.name).call(null, tree, options);
};

module.exports.preprocessCss = function(tree, inputPath, outputPath, options) {
  var plugins = options.registry.load('css');

  if (plugins.length === 0) {
    var Funnel = require('broccoli-funnel');

    return new Funnel(tree, {
      srcDir: inputPath,

      getDestinationPath: function(relativePath) {
        if (options.outputPaths) {
          // options.outputPaths is not present when compiling
          // an addon's styles
          var path = relativePath.replace(/\.css$/, '');

          // is a rename rule present?
          if (options.outputPaths[path]) {
            return options.outputPaths[path];
          }
        }

        return outputPath + '/' + relativePath;
      }
    });
  }

  return processPlugins(plugins, arguments);
};

module.exports.preprocessTemplates = function(/* tree */) {
  var options = arguments[arguments.length - 1];
  var plugins = options.registry.load('template');

  debug('plugins found for templates: %s', plugins.map(function(p) { return p.name; }));

  if (plugins.length === 0) {
    throw new Error('Missing template processor');
  }

  return processPlugins(plugins, arguments);
};

module.exports.preprocessJs = function(/* tree, inputPath, outputPath, options */) {
  var options = arguments[arguments.length - 1];
  var plugins = options.registry.load('js');
  var tree    = arguments[0];

  if (plugins.length === 0) { return tree; }

  return processPlugins(plugins, arguments);
};

function processPlugins(plugins, args) {
  args = Array.prototype.slice.call(args);
  var tree = args.shift();

  plugins.forEach(function(plugin) {
    debug('processing %s', plugin.name);
    tree = plugin.toTree.apply(plugin, [tree].concat(args));
  });

  return tree;
}
