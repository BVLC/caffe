'use strict';

var expect              = require('chai').expect;
var getPackageBaseName  = require('../../../lib/utilities/get-package-base-name');

describe('getPackageBaseName', function() {
  it('should return the full package name if it is unscoped', function() {
    expect(getPackageBaseName('my-addon')).to.equal('my-addon');
  });

  it('should return the package name without its scope', function() {
    expect(getPackageBaseName('@scope/my-addon')).to.equal('my-addon');
  });
});
