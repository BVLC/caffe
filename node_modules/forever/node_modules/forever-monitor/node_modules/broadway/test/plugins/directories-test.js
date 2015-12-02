/*
 * directories-test.js: Tests for working with directories in broadway.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */
 
var assert = require('assert'),
    fs = require('fs'),
    path = require('path'),
    vows = require('vows'),
    broadway = require('../../lib/broadway'),
    macros = require('../helpers/macros');

var fixturesDir   = path.join(__dirname, '..', 'fixtures'),
    emptyAppDir   = path.join(fixturesDir, 'empty-app'),
    emptyAppFile  = path.join(fixturesDir, 'sample-app.json'),
    appConfig     = JSON.parse(fs.readFileSync(emptyAppFile, 'utf8'));;

vows.describe('broadway/plugins/directories').addBatch({
  "Using the config plugin": {
    "extending an application": macros.shouldExtend(new broadway.App({
      root: emptyAppDir,
      directories: appConfig.directories
    }), 'directories', {})
  }
}).export(module);