/*
 * helpers.js: Test helpers for nconf.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */
 
var assert = require('assert'),
    spawn = require('child_process').spawn,
    util = require('util'),
    fs = require('fs'),
    path = require('path'),
    nconf = require('../lib/nconf');

exports.assertMerged = function (err, merged) {
  merged = merged instanceof nconf.Provider 
    ? merged.store.store
    : merged;
    
  assert.isNull(err);
  assert.isObject(merged);
  assert.isTrue(merged.apples);
  assert.isTrue(merged.bananas);
  assert.isObject(merged.candy);
  assert.isTrue(merged.candy.something1);
  assert.isTrue(merged.candy.something2);
  assert.isTrue(merged.candy.something3);
  assert.isTrue(merged.candy.something4);
  assert.isTrue(merged.dates);
  assert.isTrue(merged.elderberries);
};

exports.assertSystemConf = function (options) {
  return {
    topic: function () {
      var env = null;
      
      if (options.env) {
        env = {}
        Object.keys(process.env).forEach(function (key) {
          env[key] = process.env[key];
        });
        
        Object.keys(options.env).forEach(function (key) {
          env[key] = options.env[key];
        });
      }
      
      var child = spawn('node', [options.script].concat(options.argv), { env: env });
      child.stdout.once('data', this.callback.bind(this, null));
    },
    "should respond with the value passed into the script": function (_, data) {
      assert.equal(data.toString(), 'foobar');
    }
  }
}

// copy a file
exports.cp = function (from, to, callback) {
  fs.readFile(from, function (err, data) {
    if (err) return callback(err);
    fs.writeFile(to, data, callback);
  });
};

exports.fixture = function (file) {
  return path.join(__dirname, 'fixtures', file);
};