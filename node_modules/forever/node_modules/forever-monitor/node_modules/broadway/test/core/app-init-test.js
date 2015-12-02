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
  "An initialized instance of broadway.App with three plugins": {
    topic: function () {
      var app = new broadway.App(),
          that = this,
          three;

      that.init = [];

      three = {
        name: 'three',
        init: function (cb) {
          process.nextTick(function () {
            that.init.push('three');
            cb();
          })
        }
      };

      // First plugin. Includes an init step.
      app.use({
        attach: function () {
          this.place = 'rackspace';
        },

        init: function (cb) {
          var self = this;

          // a nextTick isn't technically necessary, but it does make this
          // purely async.
          process.nextTick(function () {
            that.init.push('one');
            self.letsGo = function () {
              return 'Let\'s go to '+self.place+'!';
            }

            cb();
          });
        }
      });

      // Second plugin. Only involves an "attach".
      app.use({
        attach: function () {
          this.oneup = function (n) {
            n++;
            return n;
          }
        }
      });
      
      // Third pluging. Only involves an "init".
      app.use(three);
      
      // Attempt to use it again. This should not invoke `init()` twice
      app.use(three);

      // Remove the plugin and use it again. This should not invoke `init()` twice
      app.remove(three);
      app.use(three);
      
      // Removing a plugin which was never added should not affect the initlist
      app.remove({
        name: 'foo'
      });
      
      app.init(function (err) {
        that.callback(err, app);
      });
    },
    "shouldn't throw an error": function (err, app) {
      assert.ok(!err);
    },
    "should have all its methods attached/defined": function (err, app) {
      assert.ok(app.place);
      assert.isFunction(app.oneup);
      assert.isFunction(app.letsGo);
      assert.equal(2, app.oneup(1));
      assert.equal(app.letsGo(), 'Let\'s go to rackspace!');

      //
      // This is intentional. The second plugin does not invoke `init`.
      //
      assert.deepEqual(this.init, ['one', 'three']);
    },
  }
}).export(module);
