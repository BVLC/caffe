/*
 * nconf-nested-env.js: Test fixture for env with nested keys.
 *
 * (C) 2012, Nodejitsu Inc.
 * (C) 2012, Michael Hart
 *
 */
 
var nconf = require('../../../lib/nconf').env('_');

process.stdout.write(nconf.get('SOME:THING'));
