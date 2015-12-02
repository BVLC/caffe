/*
 * provider-argv.js: Test fixture for using optimist defaults with nconf.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */
 
var nconf = require('../../../lib/nconf');

var provider = new (nconf.Provider)().argv();

process.stdout.write(provider.get('something'));