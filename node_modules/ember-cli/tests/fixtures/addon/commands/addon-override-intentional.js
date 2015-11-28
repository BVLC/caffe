var Command = require('../../../../lib/models/command');

function Addon() {
  this.name = "Other Ember CLI Addon Command To Test Intentional Core Command Override"
  return this;
}

Addon.prototype.includedCommands = function() {
  var cmd = Command.extend({
    name: 'serve',
    description: 'this intentionally overrides the core command'
  });
  cmd.overrideCore = true

  return {
    'InitOverrideIntentional': cmd
  };
}

module.exports = Addon;
