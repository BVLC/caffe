/*
 * config-test.js: Tests for the broadway config plugin
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */
 
var vows = require('vows'),
    events = require('eventemitter2'),
    assert = require('../helpers/assert'),
    macros = require('../helpers/macros'),
    broadway = require('../../lib/broadway');
    
vows.describe('broadway/plugins/config').addBatch({
  "Using the config plugin": {
    "extending an application": macros.shouldExtend('config')
  }
}).export(module);