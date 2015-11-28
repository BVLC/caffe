'use strict';

var assign         = require('lodash/object/assign');
var expect         = require('chai').expect;
var PluginRegistry = require('../../');

var pkg, registry, app;

describe('Plugin Loader', function() {

  beforeEach(function() {
    pkg = {
      dependencies: {
        'broccoli-emblem': 'latest'
      },
      devDependencies: {
        'broccoli-sass': 'latest',
        'broccoli-coffee': 'latest'
      }
    };

    app = { name: 'some-application-name' };
    registry = new PluginRegistry(assign(pkg.devDependencies, pkg.dependencies), app);
    registry.add('css', 'broccoli-sass', ['scss', 'sass']);
    registry.add('css', 'broccoli-ruby-sass', ['scss', 'sass']);
  });

  it('returns array of one plugin when only one', function() {
    var plugins = registry.load('css');

    expect(plugins.length).to.equal(1);
    expect(plugins[0].name).to.equal('broccoli-sass');
  });

  it('returns the correct list of plugins when there are more than one', function() {
    registry.availablePlugins['broccoli-ruby-sass'] = 'latest';
    var plugins = registry.load('css');

    expect(plugins.length).to.equal(2);
    expect(plugins[0].name).to.equal('broccoli-sass');
    expect(plugins[1].name).to.equal('broccoli-ruby-sass');
  });

  it('returns plugin of the correct type', function() {
    registry.add('js', 'broccoli-coffee');
    var plugins = registry.load('js');

    expect(plugins.length).to.equal(1);
    expect(plugins[0].name).to.equal('broccoli-coffee');
  });

  it('returns plugin that was in dependencies', function() {
    registry.add('template', 'broccoli-emblem');
    var plugins = registry.load('template');
    expect(plugins[0].name).to.equal('broccoli-emblem');
  });

  it('returns null when no plugin available for type', function() {
    registry.add('blah', 'not-available');
    var plugins = registry.load('blah');
    expect(plugins.length).to.equal(0);
  });

  it('returns the configured extension for the plugin', function() {
    registry.add('css', 'broccoli-less-single', 'less');
    registry.availablePlugins = { 'broccoli-less-single': 'latest' };
    var plugins = registry.load('css');

    expect(plugins[0].ext).to.equal('less');
  });

  it('can specify fallback extensions', function() {
    registry.availablePlugins = { 'broccoli-ruby-sass': 'latest' };
    var plugins = registry.load('css');
    var plugin  = plugins[0];

    expect(plugin.ext[0]).to.equal('scss');
    expect(plugin.ext[1]).to.equal('sass');
  });

  it('provides the application name to each plugin', function() {
    registry.add('js', 'broccoli-coffee');
    var plugins = registry.load('js');

    expect(plugins[0].applicationName).to.equal('some-application-name');
  });

  it('adds a plugin directly if it is provided', function() {
    var randomPlugin = {name: 'Awesome!'};

    registry.add('js', randomPlugin);
    var registered = registry.registry['js'];

    expect(registered[0]).to.equal(randomPlugin);
  });

  it('returns plugins added manually even if not present in package deps', function() {
    var randomPlugin = {name: 'Awesome!'};

    registry.add('foo', randomPlugin);
    var plugins = registry.load('foo');

    expect(plugins[0]).to.equal(randomPlugin);
  });

  describe('extensionsForType', function() {
    it('includes all extensions for the given type', function() {

      var extensions = registry.extensionsForType('css');

      expect(extensions).to.deep.equal(['css', 'scss', 'sass']);
    });

    it('can handle mixed array and non-array extensions', function() {
      registry.add('css', 'broccoli-foo', 'foo');
      var extensions = registry.extensionsForType('css');

      expect(extensions).to.deep.equal(['css', 'scss', 'sass', 'foo']);
    });
  });

  describe('adds a plugin directly if it is provided', function() {
    it('returns an empty array if called on an unknown type', function() {
      expect(registry.registeredForType('foo')).to.deep.equal([]);
    });

    it('returns the current array if type is found', function() {
      var fooArray = [ 'something', 'else' ];

      registry.registry['foo'] = fooArray;

      expect(registry.registeredForType('foo')).to.deep.equal(fooArray);
    });
  });

  it('allows removal of a specified plugin', function() {
    registry.availablePlugins['broccoli-ruby-sass'] = 'latest';
    registry.remove('css', 'broccoli-sass');

    var plugins = registry.load('css');
    expect(plugins.length).to.equal(1);
    expect(plugins[0].name).to.equal('broccoli-ruby-sass');
  });

  it('allows removal of plugin added as instantiated objects', function() {
    var randomPlugin, plugins;

    randomPlugin = {name: 'Awesome!'};
    registry.add('foo', randomPlugin);

    plugins = registry.load('foo');
    expect(plugins[0]).to.equal(randomPlugin); // precondition

    registry.remove('foo', randomPlugin);

    plugins = registry.load('foo');
    expect(plugins.length).to.equal(0);
  });

  it('an unfound item does not affect the registered list', function() {
    var plugins;

    function pluginNames(plugins) {
      return plugins.map(function(p) { return p.name; });
    }

    registry.availablePlugins['blah-zorz'] = 'latest';
    registry.availablePlugins['blammo'] = 'latest';

    registry.add('foo', 'blah-zorz', 'zorz');
    registry.add('foo', 'blammo', 'blam');

    plugins = registry.load('foo');

    expect(pluginNames(plugins)).to.eql(['blah-zorz', 'blammo']); // precondition

    registry.remove('foo', 'nothing I know');
    plugins = registry.load('foo');

    expect(pluginNames(plugins)).to.eql(['blah-zorz', 'blammo']);

    registry.remove('foo', 'blah-zorz');
    plugins = registry.load('foo');

    expect(pluginNames(plugins)).to.eql(['blammo']);
  });
});
