/*
 * nconf-hierarchical-load-save.js: Test fixture for using optimist, envvars and a file store with nconf.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */
 
var fs = require('fs'),
    path = require('path'),
    nconf = require('../../../lib/nconf');

//
// Setup nconf to use (in-order): 
//   1. Command-line arguments
//   2. Environment variables
//   3. A file located at 'path/to/config.json'
//
nconf.argv()
     .env()
     .file({ file: path.join(__dirname, '..', 'load-save.json') });

//
// Set a few variables on `nconf`.
//
nconf.set('database:host', '127.0.0.1');
nconf.set('database:port', 5984);

process.stdout.write(nconf.get('foo'));
//
// Save the configuration object to disk
//
nconf.save();