/*
 * directories-test.js: Tests for working with directories in broadway.
 *
 * (C) 2011, Nodejitsu Inc.
 * MIT LICENSE
 *
 */
 
var assert = require('assert'),
    fs = require('fs'),
    path = require('path'),
    vows = require('vows'),
    broadway = require('../../lib/broadway');

var fixturesDir   = path.join(__dirname, '..', 'fixtures'),
    emptyAppDir   = path.join(fixturesDir, 'empty-app'),
    emptyAppFile  = path.join(fixturesDir, 'sample-app.json'),
    appConfig     = JSON.parse(fs.readFileSync(emptyAppFile, 'utf8')),
    directories   = appConfig.directories;

vows.describe('broadway/common/directories').addBatch({
  "When using broadway.common.directories": {
    "it should have the correct methods defined": function () {
      assert.isObject(broadway.common.directories);
      assert.isFunction(broadway.common.directories.create);
      assert.isFunction(broadway.common.directories.remove);
    },
    "the normalize() method should correctly modify a set of directories": function () {
      directories = broadway.common.directories.normalize({'#ROOT': emptyAppDir}, directories);
      
      Object.keys(directories).forEach(function (key) {
        assert.isTrue(directories[key].indexOf(emptyAppDir) !== -1);
      });
    },
    "the create() method": {
      topic: function () {
        broadway.common.directories.create(directories, this.callback);
      },
      "should create the specified directories": function (err, dirs) {
        assert.isTrue(!err);
        
        dirs.forEach(function (dir) {
          assert.isTrue((fs.existsSync || path.existsSync)(dir));
        });
      },
      "the destroy() method": {
        topic: function () {
          broadway.common.directories.remove(directories, this.callback);
        },
        "should remove the specified directories": function (err, dirs) {
          assert.isTrue(!err);

          dirs.forEach(function (dir) {
            assert.isFalse((fs.existsSync || path.existsSync)(dir));
          });
        }
      }
    }
  }
}).export(module);
