/*
 * path-test.js: Tests for the core `.path()` method.
 *
 * (C) 2011, Charlie Robbins, Paolo Fragomeni, & the Contributors.
 * MIT LICENSE
 *
 */

var assert = require('assert'),
    vows = require('vows'),
    director = require('../../../lib/director');

vows.describe('director/core/path').addBatch({
  "An instance of director.Router": {
    topic: function () {
      var that = this;
      that.matched = {};
      that.matched['foo'] = [];
      that.matched['newyork'] = [];

      var router = new director.Router({
        '/foo': function () { that.matched['foo'].push('foo'); }
      });
      return router;
    },
    "the path() method": {
      "should create the correct nested routing table": function (router) {
        var that = this;
        router.path('/regions', function () {
          this.on('/:state', function(country) {
            that.matched['newyork'].push('new york');
          });
        });

        assert.isFunction(router.routes.foo.on);
        assert.isObject(router.routes.regions);
        assert.isFunction(router.routes.regions['([._a-zA-Z0-9-%()]+)'].on);
      },
      "should dispatch the function correctly": function (router) {
        router.dispatch('on', '/regions/newyork')
        router.dispatch('on', '/foo');
        assert.equal(this.matched['foo'].length, 1);
        assert.equal(this.matched['newyork'].length, 1);
        assert.equal(this.matched['foo'][0], 'foo');
        assert.equal(this.matched['newyork'][0], 'new york');
      }
    }
  }
}).export(module);
