'use strict';

var broccoli = require('broccoli');
var walkSync = require('walk-sync');
var expect   = require('chai').expect;
var Reexporter = require('../../../lib/utilities/reexport');

describe('reexport-tree', function() {
  var builder;

  afterEach(function() {
    if (builder) {
      return builder.cleanup();
    }
  });

  it('rebuilds without error', function() {
    var tree = new Reexporter('something', 'file.js');

    builder = new broccoli.Builder(tree);
      return builder.build()
        .then(function() {
          return builder.build();
        })
        .then(function(results) {
          var outputPath = results.directory;

          var expected = [
            'reexports/',
            'reexports/file.js'
          ];

          expect(walkSync(outputPath)).to.eql(expected);
        });
  });
});
