/*
 * complete-test.js: Complete test for multiple stores.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */

var fs = require('fs'),
    path = require('path'),
    vows = require('vows'),
    assert = require('assert'),
    nconf = require('../lib/nconf'),
    data = require('./fixtures/data').data,
    helpers = require('./helpers');

var completeTest = helpers.fixture('complete-test.json'),
    complete = helpers.fixture('complete.json');

vows.describe('nconf/multiple-stores').addBatch({
  "When using the nconf with multiple providers": {
    topic: function () {
      var that = this;
      helpers.cp(complete, completeTest, function () {
        nconf.env();
        nconf.file({ file: completeTest });
        nconf.use('argv', { type: 'literal', store: data });
        that.callback();
      });
    },
    "should have the correct `stores`": function () {
      assert.isObject(nconf.stores.env);
      assert.isObject(nconf.stores.argv);
      assert.isObject(nconf.stores.file);
    },
    "env vars": {
      "are present": function () {
        Object.keys(process.env).forEach(function (key) {
          assert.equal(nconf.get(key), process.env[key]);
        });
      }
    },
    "json vars": {
      topic: function () {
        fs.readFile(complete, 'utf8', this.callback);
      },
      "are present": function (err, data) {
        assert.isNull(err);
        data = JSON.parse(data);
        Object.keys(data).forEach(function (key) {
          assert.deepEqual(nconf.get(key), data[key]);
        });
      }
    },
    "literal vars": {
      "are present": function () {
        Object.keys(data).forEach(function (key) {
          assert.deepEqual(nconf.get(key), data[key]);
        });
      }
    },
    "and saving *synchronously*": {
      topic: function () {
        nconf.set('weebls', 'stuff');
        return nconf.save();
      },
      "correct return value": function (topic) {
        Object.keys(topic).forEach(function (key) {
          assert.deepEqual(topic[key], nconf.get(key));
        });
      },
      "the file": {
        topic: function () {
          fs.readFile(completeTest, 'utf8', this.callback);
        },
        "saved correctly": function (err, data) {
          data = JSON.parse(data);
          Object.keys(data).forEach(function (key) {
            assert.deepEqual(data[key], nconf.get(key));
          });
          assert.equal(nconf.get('weebls'), 'stuff');
        }
      }
    },
    teardown: function () {
      // remove the file so that we can test saving it async
      fs.unlinkSync(completeTest);
    }
  }
}).addBatch({
  // Threw this in it's own batch to make sure it's run separately from the
  // sync check
  "When using the nconf with multiple providers": {
    "and saving *asynchronously*": {
      topic: function () {
        nconf.set('weebls', 'crap');
        nconf.save(this.callback);
      },
      "correct return value": function (err, data) {
        assert.isNull(err);
        Object.keys(data).forEach(function (key) {
          assert.deepEqual(data[key], nconf.get(key));
        });
      },
      "the file": {
        topic: function () {
          fs.readFile(completeTest, 'utf8', this.callback);
        },
        "saved correctly": function (err, data) {
          assert.isNull(err);
          data = JSON.parse(data);
          Object.keys(data).forEach(function (key) {
            assert.deepEqual(nconf.get(key), data[key]);
          });
          assert.equal(nconf.get('weebls'), 'crap');
        }
      }
    },
    teardown: function () {
      fs.unlinkSync(completeTest);
      nconf.remove('file');
      nconf.remove('memory');
      nconf.remove('argv');
      nconf.remove('env');
    }
  }
}).export(module);