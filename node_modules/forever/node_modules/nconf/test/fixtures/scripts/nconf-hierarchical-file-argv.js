/*
 * nconf-hierarchical-file-argv.js: Test fixture for using optimist defaults and a file store with nconf.
 *
 * (C) 2011, Nodejitsu Inc.
 * (C) 2011, Sander Tolsma
 *
 */
 
var path = require('path'),
    nconf = require('../../../lib/nconf');

nconf.argv();
nconf.add('file', {
  file: path.join(__dirname, '../hierarchy/hierarchical.json')
});

process.stdout.write(nconf.get('something') || 'undefined');