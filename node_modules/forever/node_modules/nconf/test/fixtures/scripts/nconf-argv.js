/*
 * default-argv.js: Test fixture for using optimist defaults with nconf.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */
 
var nconf = require('../../../lib/nconf').argv().env();

process.stdout.write(nconf.get('something'));