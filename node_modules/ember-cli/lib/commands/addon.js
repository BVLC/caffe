'use strict';

var NewCommand = require('./new');

module.exports = NewCommand.extend({
  name: 'addon',
  description: 'Generates a new folder structure for building an addon, complete with test harness.',

  availableOptions: [
    { name: 'dry-run',    type: Boolean, default: false,   aliases: ['d'] },
    { name: 'verbose',    type: Boolean, default: false,   aliases: ['v'] },
    { name: 'blueprint',  type: String,  default: 'addon', aliases: ['b'] },
    { name: 'skip-npm',   type: Boolean, default: false,   aliases: ['sn'] },
    { name: 'skip-bower', type: Boolean, default: false,   aliases: ['sb'] },
    { name: 'skip-git',   type: Boolean, default: false,   aliases: ['sg'] },
  ],

  anonymousOptions: [
    '<addon-name>'
  ]
});
