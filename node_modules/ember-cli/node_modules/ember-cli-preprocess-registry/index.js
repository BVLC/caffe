'use strict';

var Plugin           = require('./lib/plugin');
var StylePlugin      = require('./lib/style-plugin');
var TemplatePlugin   = require('./lib/template-plugin');
var JavascriptPlugin = require('./lib/javascript-plugin');
var debug            = require('debug')('ember-cli:registry');

function Registry(plugins, app) {
  this.registry = {
    js: [],
    css: [],
    'minify-css': [],
    template: []
  };

  this.instantiatedPlugins = [];
  this.availablePlugins = plugins;
  this.app = app;
  this.pluginTypes = {
    'js': JavascriptPlugin,
    'css': StylePlugin,
    'template': TemplatePlugin
  };
}

module.exports = Registry;

Registry.prototype.extensionsForType = function(type) {
  var registered = this.registeredForType(type);

  var extensions =  registered.reduce(function(memo, plugin) {
    return memo.concat(plugin.ext);
  }, [type]);

  extensions = require('lodash/array/uniq')(extensions);

  debug('extensions for type %s: %s', type, extensions);

  return extensions;
};

Registry.prototype.load = function(type) {
  var knownPlugins = this.registeredForType(type);
  var plugins = knownPlugins.map(function(plugin) {
    if(this.instantiatedPlugins.indexOf(plugin) > -1 || this.availablePlugins.hasOwnProperty(plugin.name)) {
      return plugin;
    }
  }.bind(this))
  .filter(Boolean);

  debug('loading %s: available plugins %s; found plugins %s;', type, knownPlugins.map(function(p) { return p.name; }), plugins.map(function(p) { return p.name; }));

  return plugins;
};

Registry.prototype.registeredForType = function(type) {
  return this.registry[type] = this.registry[type] || [];
};

Registry.prototype.add = function(type, name, extension, options) {
  var registered = this.registeredForType(type);
  var plugin, PluginType;

  // plugin is being added directly do not instantiate it
  if (typeof name === 'object') {
    plugin = name;
    this.instantiatedPlugins.push(plugin);
  } else {
    PluginType = this.pluginTypes[type] || Plugin;
    options = options || {};
    options.applicationName = this.app.name;
    options.app = this.app;

    plugin = new PluginType(name, extension, options);
  }

  debug('add type: %s, name: %s, extension:%s, options:%s', type, plugin.name, plugin.ext, options);

  registered.push(plugin);
};

Registry.prototype.remove = function(type /* name */) {
  var registered = this.registeredForType(type);
  var registeredIndex, name;

  if (typeof arguments[1] === 'object') {
    name = arguments[1].name;
  } else {
    name = arguments[1];
  }

  debug('remove type: %s, name: %s', type, name);

  for (var i = 0, l = registered.length; i < l; i++) {
    if (registered[i].name === name) {
      registeredIndex = i;
    }
  }

  var plugin = registered[registeredIndex];
  var instantiatedPluginIndex = this.instantiatedPlugins.indexOf(plugin);

  if (instantiatedPluginIndex > -1) {
    this.instantiatedPlugins.splice(instantiatedPluginIndex, 1);
  }

  if (registeredIndex !== undefined && registeredIndex > -1) {
    registered.splice(registeredIndex, 1);
  }
};
