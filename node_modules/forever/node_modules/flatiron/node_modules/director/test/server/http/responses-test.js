/*
 * responses-test.js: Tests for HTTP responses.
 *
 * (C) 2011, Charlie Robbins, Paolo Fragomeni, & the Contributors.
 * MIT LICENSE
 *
 */

var assert = require('assert'),
    vows = require('vows'),
    director = require('../../../lib/director');

vows.describe('director/http/responses').addBatch({
  "When using director.http": {
    "it should have the relevant responses defined": function () {
      Object.keys(require('../../../lib/director/http/responses')).forEach(function (name) {
        assert.isFunction(director.http[name]);
      });
    }
  }
}).export(module);
