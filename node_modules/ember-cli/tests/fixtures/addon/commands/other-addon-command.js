function Addon() {
  this.name = "Other Ember CLI Addon Command Test"
  return this;
}

Addon.prototype.includedCommands = function() {
  return {
    'OtherAddonCommand': {
      name: 'other-addon-command',
      aliases: ['oac']
    }
  };
}

module.exports = Addon;
