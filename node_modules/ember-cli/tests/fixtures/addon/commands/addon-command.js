function Addon() {
  this.name = "Ember CLI Addon Command Test"
  return this;
}

Addon.prototype.includedCommands = function() {
  return {
    'AddonCommand': {
      name: 'addon-command',
      aliases: ['ac']
    }
  };
}

module.exports = Addon;
