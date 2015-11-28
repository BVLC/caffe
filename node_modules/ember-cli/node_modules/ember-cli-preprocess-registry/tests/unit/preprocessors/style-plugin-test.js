'use strict';

var expect      = require('chai').expect;
var StylePlugin = require('../../../lib/style-plugin');

describe('Style Plugin', function(){
  describe('constructor', function(){
    var plugin;
    var options;
    before(function(){
      options = {
        paths: ['some/path'],
        registry: 'some/registry',
        applicationName: 'some/application'
      };
      plugin = new StylePlugin('california-stylesheets', 'cass', options);
    });
    it('sets type', function(){
      expect(plugin.type).to.equal('css');
    });
    it('sets name', function(){
      expect(plugin.name).to.equal('california-stylesheets');
    });
    it('sets ext', function(){
      expect(plugin.ext).to.equal('cass');
    });
    it('sets options', function(){
      expect(plugin.options).to.equal(options);
    });
    it('sets registry', function(){
      expect(plugin.registry).to.equal('some/registry');
    });
    it('sets applicationName', function(){
      expect(plugin.applicationName).to.equal('some/application');
    });
  });
});

