/*
 * common.js: Tests for common utility function in nconf.
 *
 * (C) 2011, Nodejitsu Inc.
 *
 */

var fs = require('fs'),
    path = require('path'),
    vows = require('vows'),
    assert = require('assert'),
    helpers = require('./helpers'),
    nconf = require('../lib/nconf');

var mergeDir = path.join(__dirname, 'fixtures', 'merge'),
    files = fs.readdirSync(mergeDir).map(function (f) { return path.join(mergeDir, f) });

vows.describe('nconf/common').addBatch({
  "Using nconf.common module": {
    "the loadFiles() method": {
      topic: function () {
        nconf.loadFiles(files, this.callback);
      },
      "should merge the files correctly": helpers.assertMerged
    },
    "the loadFilesSync() method": {
      "should merge the files correctly": function () {
        helpers.assertMerged(null, nconf.loadFilesSync(files));
      }
    }
  }
}).export(module);