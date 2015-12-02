/*
 * grunt-cli
 * http://gruntjs.com/
 *
 * Copyright (c) 2012 Tyler Kellen, contributors
 * Licensed under the MIT license.
 * https://github.com/gruntjs/grunt-init/blob/master/LICENSE-MIT
 */

'use strict';

// External lib.
var nopt = require('nopt');

// CLI options we care about.
exports.known = {help: Boolean, version: Boolean, completion: String};
exports.aliases = {h: '--help', V: '--version', v: '--verbose'};

// Parse them and return an options object.
Object.defineProperty(exports, 'options', {
  get: function() {
    return nopt(exports.known, exports.aliases, process.argv, 2);
  }
});
