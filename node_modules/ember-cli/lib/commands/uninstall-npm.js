'use strict';

var Command     = require('../models/command');
var SilentError = require('silent-error');
var Promise     = require('../ext/promise');

module.exports = Command.extend({
  name: 'uninstall:npm',
  description: 'Npm package uninstall are now managed by the user.',
  works: 'insideProject',
  skipHelp: true,

  anonymousOptions: [
    '<package-names...>'
  ],

  run: function() {
    var err = 'This command has been removed. Please use `npm uninstall ';
    err += '<packageName> --save-dev` instead.';
    return Promise.reject(new SilentError(err));
  }
});
