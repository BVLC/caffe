/*
 * file-store-test.js: Tests for the nconf File store.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */

var fs = require('fs'),
    path = require('path'),
    vows = require('vows'),
    assert = require('assert'),
    nconf = require('../lib/nconf'),
    data = require('./fixtures/data').data;

vows.describe('nconf').addBatch({
  "When using the nconf": {
    "should have the correct methods set": function () {
      assert.isFunction(nconf.key);
      assert.isFunction(nconf.path);
      assert.isFunction(nconf.use);
      assert.isFunction(nconf.get);
      assert.isFunction(nconf.set);
      assert.isFunction(nconf.clear);
      assert.isFunction(nconf.load);
      assert.isFunction(nconf.save);
      assert.isFunction(nconf.reset);
    },
    "the use() method": {
      "should instaniate the correct store": function () {
        nconf.use('memory');
        nconf.load();
        assert.instanceOf(nconf.stores['memory'], nconf.Memory);
      }
    },
    "it should": {
      topic: function () {
        fs.readFile(path.join(__dirname, '..', 'package.json'), this.callback);
      },
      "have the correct version set": function (err, data) {
        assert.isNull(err);
        data = JSON.parse(data.toString());
        assert.equal(nconf.version, data.version);
      }
    }
  }
}).addBatch({
  "When using the nconf": {
    "with the memory store": {
      "the set() method": {
        "should respond with true": function () {
          assert.isTrue(nconf.set('foo:bar:bazz', 'buzz'));
        }
      },
      "the get() method": {
        "without a callback": {
          "should respond with the correct value": function () {
            assert.equal(nconf.get('foo:bar:bazz'), 'buzz');
          }
        },
        "with a callback": {
          topic: function () {
            nconf.get('foo:bar:bazz', this.callback);
          },
          "should respond with the correct value": function (err, value) {
            assert.equal(value, 'buzz');
          }
        }
      }
    }
  }
}).addBatch({
  "When using the nconf": {
    "with the memory store": {
      "the get() method": {
        "should respond allow access to the root": function () {
          assert(nconf.get(null));
          assert(nconf.get(undefined));
          assert(nconf.get());
        }
      },
      "the set() method": {
        "should respond allow access to the root and complain about non-objects": function () {
          assert(!nconf.set(null, null));
          assert(!nconf.set(null, undefined));
          assert(!nconf.set(null));
          assert(!nconf.set(null, ''));
          assert(!nconf.set(null, 1));
          var original = nconf.get();
          assert(nconf.set(null, nconf.get()));
          assert.notEqual(nconf.get(), original);
          assert.deepEqual(nconf.get(), original)
        }
      }
    }
  }
}).addBatch({
  "When using nconf": {
    "with the memory store": {
      "the clear() method": {
        "should respond with the true": function () {
          assert.equal(nconf.get('foo:bar:bazz'), 'buzz');
          assert.isTrue(nconf.clear('foo:bar:bazz'));
          assert.isTrue(typeof nconf.get('foo:bar:bazz') === 'undefined');
        }
      },
      "the load() method": {
        "without a callback": {
          "should respond with the merged store": function () {
            assert.deepEqual(nconf.load(), {
              title: 'My specific title', 
              color: 'green',
              movie: 'Kill Bill' 
            });
          }
        },
        "with a callback": {
          topic: function () {
            nconf.load(this.callback.bind(null, null)); 
          },
          "should respond with the merged store": function (ign, err, store) {
            assert.isNull(err);
            assert.deepEqual(store, {
              title: 'My specific title', 
              color: 'green',
              movie: 'Kill Bill' 
            });
          }
        }
      }
    }
  }
}).export(module);
