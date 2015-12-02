/*
 * grunt-cli
 * http://gruntjs.com/
 *
 * Copyright (c) 2012 Tyler Kellen, contributors
 * Licensed under the MIT license.
 * https://github.com/gruntjs/grunt-init/blob/master/LICENSE-MIT
 */

'use strict';

// Project metadata.
var pkg = require('../package.json');

// Display grunt-cli version.
exports.version = function() {
  console.log('grunt-cli v' + pkg.version);
};

// Show help, then exit with a message and error code.
exports.fatal = function(msg, code) {
  exports.helpHeader();
  console.log('Fatal error: ' + msg);
  console.log('');
  exports.helpFooter();
  process.exit(code);
};

// Show help and exit.
exports.help = function() {
  exports.helpHeader();
  exports.helpFooter();
  process.exit();
};

// Help header.
exports.helpHeader = function() {
  console.log('grunt-cli: ' + pkg.description + ' (v' + pkg.version + ')');
  console.log('');
};

// Help footer.
exports.helpFooter = function() {
  [
    'If you\'re seeing this message, either a Gruntfile wasn\'t found or grunt',
    'hasn\'t been installed locally to your project. For more information about',
    'installing and configuring grunt, please see the Getting Started guide:',
    '',
    'http://gruntjs.com/getting-started',
  ].forEach(function(str) { console.log(str); });
};
