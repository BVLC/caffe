'use strict';

var Command     = require('../models/command');
var SilentError = require('silent-error');
var Promise     = require('../ext/promise');

module.exports = Command.extend({
  name: 'install:npm',
  description: 'Npm package installs are now managed by the user.',
  works: 'insideProject',
  skipHelp: true,

  anonymousOptions: [
    '<package-names...>'
  ],

  run: function() {
    var err = 'This command has been removed. Please use `npm install ';
    err += '<packageName> --save-dev --save-exact` instead.';
    return Promise.reject(new SilentError(err));
  }
});
