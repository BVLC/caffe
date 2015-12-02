/*
 * app-test.js: Tests for core App methods and configuration.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */
 
var events = require('eventemitter2'),
    vows = require('vows'),
    assert = require('../helpers/assert'),
    broadway = require('../../lib/broadway');
    
vows.describe('broadway/app').addBatch({
  "An instance of broadway.App": {
    topic: new broadway.App(),
    "should have the correct properties and methods": function (app) {
      //
      // Instance
      //
      assert.isObject(app);
      assert.instanceOf(app, events.EventEmitter2);
      assert.instanceOf(app, broadway.App);
      
      //
      // Properties
      //
      assert.isObject(app.plugins);
      assert.isObject(app.initializers);
      assert.isFalse(!!app.initialized);
      
      //
      // Methods
      //
      assert.isFunction(app.init);
      assert.isFunction(app.use);
      assert.isFunction(app.remove);
      assert.isFunction(app.inspect);
    },
    "the init() method": {
      topic: function (app) {
        this.app = app;
        app.init(this.callback);
      },
      "should correctly setup the application state": function () {
        assert.isTrue(this.app.initialized);
        assert.isTrue(this.app.initializers['log']);
        
        assert.plugins.has.config(this.app);
        assert.plugins.has.log(this.app);
      }
    },
    "the detach() method": {
      topic: function (app) {
        app.use({ 
          name: "foo", 
          attach: function () {
            this.attached = true;
          },
          detach: function () {
            this.detached = true;
          }
        });
        app.remove("foo");
        return app;
      },
      "should correctly remove a plugin": function (app) {
        assert.isTrue(app.detached);
        assert.equal(undefined, app.plugins["foo"]);
      }
    }
  }
}).export(module);
