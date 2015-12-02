/*
 * broadway-test.js: Tests for core App methods and configuration.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */
 
var assert = require('assert'),
    events = require('eventemitter2'),
    vows = require('vows'),
    broadway = require('../../lib/broadway');
    
vows.describe('broadway').addBatch({
  "The broadway module": {
    "should have the appropriate properties and methods defined": function () {
      assert.isFunction(broadway.App);
      assert.isObject(broadway.common);
      assert.isObject(broadway.features);
      assert.isObject(broadway.plugins);
      assert.isObject(broadway.plugins.log);
      assert.isObject(broadway.plugins.config);
      assert.isObject(broadway.plugins.exceptions);
    }
  }
}).export(module);