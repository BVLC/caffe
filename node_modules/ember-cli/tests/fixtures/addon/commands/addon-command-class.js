var Command = require('../../../../lib/models/command');

function Addon() {
  this.name = "Ember CLI Addon Class Command Test"
  return this;
}

Addon.prototype.includedCommands = function() {
  return {
    'ClassAddonCommand': Command.extend({
      name: 'class-addon-command',
      aliases: ['oac']
    })
  };
}

module.exports = Addon;
