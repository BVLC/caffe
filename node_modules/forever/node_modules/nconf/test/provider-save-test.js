/*
 * provider-save-test.js: Ensures consistency for Provider `save` operations.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */

var assert = require('assert'),
    vows = require('vows'),
    nconf = require('../lib/nconf');
    
//
// Expose `nconf.Mock`
//
require('./mocks/mock-store');
    
vows.describe('nconf/provider/save').addBatch({
  "When using nconf": {
    "an instance of 'nconf.Provider'": {
      "with a Mock store": {
        topic: function () {
          return nconf.use('mock');
        },
        "the save() method": {
          topic: function () {
            var mock = nconf.stores.mock,
                that = this;
                
            mock.on('save', function () { that.saved = true });
            nconf.save(this.callback);
          },
          "should actually save before responding": function () {
            assert.isTrue(this.saved);
          }
        }
      }
    }
  }
}).export(module);