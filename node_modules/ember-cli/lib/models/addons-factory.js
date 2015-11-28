'use strict';

/**
@module ember-cli
*/

var CoreObject = require('core-object');
var DAG        = require('../utilities/DAG');
var debug      = require('debug')('ember-cli:addons-factory');

/**
  AddonsFactory is responsible for instantiating a collection of addons, in the right order.

  @class AddonsFactory
  @extends CoreObject
  @constructor
*/
function AddonsFactory(addonParent, project) {
  this.addonParent = addonParent;
  this.project = project;
}

AddonsFactory.__proto__ = CoreObject;
AddonsFactory.prototype.constructor = AddonsFactory;

AddonsFactory.prototype.initializeAddons = function(addonPackages){
  var addonParent = this.addonParent;
  var project     = this.project;
  var graph       = new DAG();
  var Addon      = require('../models/addon');
  var addonInfo, emberAddonConfig;

  debug('initializeAddons for: ', typeof addonParent.name === 'function' ? addonParent.name() : addonParent.name);
  debug('     addon names are:', Object.keys(addonPackages));

  for (var name in addonPackages) {
    addonInfo        = addonPackages[name];
    emberAddonConfig = addonInfo.pkg['ember-addon'];

    graph.addEdges(name, addonInfo, emberAddonConfig.before, emberAddonConfig.after);
  }

  var addons = [];
  graph.topsort(function (vertex) {
    var addonInfo = vertex.value;
    if (addonInfo) {
      var AddonConstructor = Addon.lookup(addonInfo);
      var addon = new AddonConstructor(addonParent, project);
      if (addon.initializeAddons) {
        addon.initializeAddons();
      } else {
        addon.addons = [];
      }
      addons.push(addon);
    }
  });

  debug(' addons ordered as:', addons.map(function(addon) {
    return addon.name;
  }));

  return addons;
};

// Export
module.exports = AddonsFactory;
