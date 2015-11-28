'use strict';

var expect       = require('chai').expect;
var versionUtils = require('../../../lib/utilities/version-utils');

describe('version-utils', function() {
  it('`isDevelopment` returns false if a release version was passed in', function() {
    expect(versionUtils.isDevelopment('0.0.5')).to.equal(false);
  });

  it('`isDevelopment` returns true if a development version was passed in', function() {
    expect(versionUtils.isDevelopment('0.0.5-master-237cc6024d')).to.equal(true);
  });
});
