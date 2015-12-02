/*
 * dispatch-test.js: Tests for the core dispatch method.
 *
 * (C) 2011, Charlie Robbins, Paolo Fragomeni, & the Contributors.
 * MIT LICENSE
 *
 */

var assert = require('assert'),
    vows = require('vows'),
    director = require('../../../lib/director');

vows.describe('director/core/dispatch').addBatch({
  "An instance of director.Router": {
    topic: function () {
      var that = this;
      that.matched = {};
      that.matched['/'] = [];
      that.matched['foo'] = [];
      that.matched['f*'] = []

      var router = new director.Router({
        '/': {
          before: function () { that.matched['/'].push('before /') },
          on: function () { that.matched['/'].push('on /') },
          after: function () { that.matched['/'].push('after /') }
        },
        '/foo': {
          before: function () { that.matched.foo.push('before foo') },
          on: function () { that.matched.foo.push('on foo') },
          after: function () { that.matched.foo.push('after foo') },
          '/bar': {
            before: function () { that.matched.foo.push('before foo bar') },
            on: function () { that.matched.foo.push('foo bar') },
            after: function () { that.matched.foo.push('after foo bar') },
            '/buzz': function () { that.matched.foo.push('foo bar buzz') }
          }
        },
        '/f*': {
          '/barbie': function () { that.matched['f*'].push('f* barbie') }
        }
      });

      router.configure({
        recurse: 'backward'
      });

      return router;
    },
    "should have the correct routing table": function (router) {
      assert.isObject(router.routes.foo);
      assert.isObject(router.routes.foo.bar);
      assert.isObject(router.routes.foo.bar.buzz);
      assert.isFunction(router.routes.foo.bar.buzz.on);
    },
    "the dispatch() method": {
      "/": function (router) {
        assert.isTrue(router.dispatch('on', '/'));
        assert.isTrue(router.dispatch('on', '/'));

        assert.equal(this.matched['/'][0], 'before /');
        assert.equal(this.matched['/'][1], 'on /');
        assert.equal(this.matched['/'][2], 'after /');
      },
      "/foo/bar/buzz": function (router) {
        assert.isTrue(router.dispatch('on', '/foo/bar/buzz'));

        assert.equal(this.matched.foo[0], 'foo bar buzz');
        assert.equal(this.matched.foo[1], 'before foo bar');
        assert.equal(this.matched.foo[2], 'foo bar');
        assert.equal(this.matched.foo[3], 'before foo');
        assert.equal(this.matched.foo[4], 'on foo');
      },
      "/foo/barbie": function (router) {
        assert.isTrue(router.dispatch('on', '/foo/barbie'));
        assert.equal(this.matched['f*'][0], 'f* barbie');
      },
      "/foo/barbie/": function (router) {
        assert.isFalse(router.dispatch('on', '/foo/barbie/'));
      },
      "/foo/BAD": function (router) {
        assert.isFalse(router.dispatch('on', '/foo/BAD'));
      },
      "/bar/bar": function (router) {
        assert.isFalse(router.dispatch('on', '/bar/bar'));
      },
      "with the strict option disabled": {
        topic: function (router) {
          return router.configure({
            recurse: 'backward',
            strict: false
          });
        },
        "should have the proper configuration set": function (router) {
          assert.isFalse(router.strict);
        },
        "/foo/barbie/": function (router) {
          assert.isTrue(router.dispatch('on', '/foo/barbie/'));
          assert.equal(this.matched['f*'][0], 'f* barbie');
        },
        "/foo/bar/buzz": function (router) {
          assert.isTrue(router.dispatch('on', '/foo/bar/buzz'));

          assert.equal(this.matched.foo[0], 'foo bar buzz');
          assert.equal(this.matched.foo[1], 'before foo bar');
          assert.equal(this.matched.foo[2], 'foo bar');
          assert.equal(this.matched.foo[3], 'before foo');
          assert.equal(this.matched.foo[4], 'on foo');
        },
      }
    }
  }
}).export(module);
