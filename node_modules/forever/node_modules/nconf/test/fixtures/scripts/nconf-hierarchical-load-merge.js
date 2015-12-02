/*
 * nconf-hierarchical-load-merge.js: Test fixture for loading and merging nested objects across stores.
 *
 * (C) 2012, Nodejitsu Inc.
 * (C) 2012, Michael Hart
 *
 */
 
var path = require('path'),
    nconf = require('../../../lib/nconf');

nconf.argv()
     .file(path.join(__dirname, '..', 'merge', 'file1.json'));

process.stdout.write(JSON.stringify({
  apples: nconf.get('apples'),
  candy: nconf.get('candy')
}));
