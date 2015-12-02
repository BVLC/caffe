var assert = require('assert'),
    vows = require('vows'),
    broadway = require('broadway'),
    flatiron = require('../');

vows.describe('flatiron').addBatch({
  'When using `flatiron`': {
    '`flatiron.plugins`': {
      topic: flatiron.plugins,
      'should contain all `broadway.plugins`': function (plugins) {
        Object.keys(broadway.plugins).forEach(function (key) {
          assert.include(plugins, key);
        });
      }
    }
  }
}).export(module);

