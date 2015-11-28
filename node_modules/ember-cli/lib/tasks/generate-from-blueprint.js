/*jshint quotmark: false*/

'use strict';

var Promise      = require('../ext/promise');
var Blueprint    = require('../models/blueprint');
var Task         = require('../models/task');
var parseOptions = require('../utilities/parse-options');
var merge        = require('lodash/object/merge');

module.exports = Task.extend({
  blueprintFunction: 'install',

  run: function(options) {
    var self = this;
    var name = options.args[0];
    var noAddonBlueprint = ['mixin'];

    var mainBlueprint  = this.lookupBlueprint(name, options.ignoreMissingMain);
    var testBlueprint  = this.lookupBlueprint(name + '-test', true);
    // lookup custom addon blueprint
    var addonBlueprint = this.lookupBlueprint(name + '-addon', true);
    // otherwise, use default addon-import

    if (noAddonBlueprint.indexOf(name) < 0 && !addonBlueprint && (mainBlueprint && mainBlueprint.supportsAddon()) && options.args[1]) {
      addonBlueprint = this.lookupBlueprint('addon-import', true);
    }

    if (options.ignoreMissingMain && !mainBlueprint) {
      return Promise.resolve();
    }

    if (options.dummy) {
      // don't install test or addon reexport for dummy
      if (this.project.isEmberCLIAddon()) {
        testBlueprint = null;
        addonBlueprint = null;
      }
    }

    var entity = {
      name: options.args[1],
      options: parseOptions(options.args.slice(2))
    };

    var blueprintOptions = {
      target: this.project.root,
      entity: entity,
      ui: this.ui,
      analytics: this.analytics,
      project: this.project,
      settings: this.settings,
      testing: this.testing,
      taskOptions: options,
      originBlueprintName: name
    };

    blueprintOptions = merge(blueprintOptions, options || {});

    return mainBlueprint[this.blueprintFunction](blueprintOptions)
      .then(function() {
        if (!testBlueprint) { return; }

        if (testBlueprint.locals === Blueprint.prototype.locals) {
          testBlueprint.locals = function(options) {
            return mainBlueprint.locals(options);
          };
        }

        var testBlueprintOptions = merge({} , blueprintOptions, { installingTest: true });

        return testBlueprint[self.blueprintFunction](testBlueprintOptions);
      })
      .then(function() {
        if (!addonBlueprint) { return; }
        if (!this.project.isEmberCLIAddon() && blueprintOptions.inRepoAddon === null) { return; }

        if (addonBlueprint.locals === Blueprint.prototype.locals) {
          addonBlueprint.locals = function(options) {
            return mainBlueprint.locals(options);
          };
        }

        var addonBlueprintOptions = merge({}, blueprintOptions, { installingAddon: true });

        return addonBlueprint[self.blueprintFunction](addonBlueprintOptions);
      }.bind(this));
  },

  lookupBlueprint: function(name, ignoreMissing) {
    return Blueprint.lookup(name, {
      paths: this.project.blueprintLookupPaths(),
      ignoreMissing: ignoreMissing
    });
  }
});
