'use strict';

var expect = require('chai').expect;
var p      = require('../../../preprocessors');

describe('setupRegistry', function() {
  var setupPreprocessorRegistryCalled, fakeAddon;

  beforeEach(function() {
    setupPreprocessorRegistryCalled = [];

    fakeAddon = {
      registry: {
        add: function() { }
      },
      addons: [],
      initializeAddons: function() { },

      setupPreprocessorRegistry: function(type, registry) {
        setupPreprocessorRegistryCalled.push([type, registry]);
      }
    };
  });

  it('calls setupPreprocessorRegistry on the provided argument if present', function() {
    p.setupRegistry(fakeAddon);

    expect(setupPreprocessorRegistryCalled.length).to.equal(1);
    expect(setupPreprocessorRegistryCalled[0][0]).to.equal('self');
    expect(setupPreprocessorRegistryCalled[0][1]).to.equal(fakeAddon.registry);
  });

  it('does not error if addon does not have `addons` property', function() {
    delete fakeAddon.addons;

    p.setupRegistry(fakeAddon);

    expect(setupPreprocessorRegistryCalled.length).to.equal(1);
  });

  describe('with nested addons', function() {
    var nestedSetupPreprocessorRegistryCalls;

    function setupPreprocessorRegistryShared(type, registry) {
      nestedSetupPreprocessorRegistryCalls.push([type, registry]);
    }

    beforeEach(function() {
      nestedSetupPreprocessorRegistryCalls = [];
    });

    it('invokes setupPreprocessorRegistry with `parent` on each addon', function() {
      fakeAddon.addons = [
        { setupPreprocessorRegistry: setupPreprocessorRegistryShared },
        { setupPreprocessorRegistry: setupPreprocessorRegistryShared },
        { setupPreprocessorRegistry: setupPreprocessorRegistryShared }
      ];

      p.setupRegistry(fakeAddon);

      expect(setupPreprocessorRegistryCalled.length).to.equal(1);
      expect(nestedSetupPreprocessorRegistryCalls.length).to.equal(3);

      nestedSetupPreprocessorRegistryCalls.forEach(function(item) {
        expect(item[0]).to.equal('parent');
        expect(item[1]).to.equal(fakeAddon.registry);
      });
    });

    it('does not error if nested addons do not have setupPreprocessorRegistry', function() {
      fakeAddon.addons = [
        { setupPreprocessorRegistry: setupPreprocessorRegistryShared },
        { },
        { }
      ];

      p.setupRegistry(fakeAddon);

      expect(nestedSetupPreprocessorRegistryCalls.length).to.equal(1);

      nestedSetupPreprocessorRegistryCalls.forEach(function(item) {
        expect(item[0]).to.equal('parent');
        expect(item[1]).to.equal(fakeAddon.registry);
      });
    });
  });
});
