'use strict';

var expect = require('chai').expect;
var PlatformChecker = require('../../../lib/utilities/platform-checker');

describe('platform-checker', function() {
  it('should return isDeprecated for Node v0.12', function() {
    expect(new PlatformChecker('v0.10.1').isDeprecated).to.be.equal(true);
    expect(new PlatformChecker('v0.10.15').isDeprecated).to.be.equal(true);
    expect(new PlatformChecker('v0.10.30').isDeprecated).to.be.equal(true);
    expect(new PlatformChecker('v0.12.0').isDeprecated).to.be.equal(false);
  });

  it('should return isUntested for Node v6', function() {
    expect(new PlatformChecker('v6.0.0').isUntested).to.be.equal(true);
    expect(new PlatformChecker('v0.12.0').isUntested).to.be.equal(false);
  });

  it('should return isValid for iojs', function() {
    expect(new PlatformChecker('v1.0.0').isValid).to.be.equal(true);
    expect(new PlatformChecker('v1.0.1').isValid).to.be.equal(true);
    expect(new PlatformChecker('v1.0.2').isValid).to.be.equal(true);
    expect(new PlatformChecker('v1.0.3').isValid).to.be.equal(true);
    expect(new PlatformChecker('v1.0.4').isValid).to.be.equal(true);
    expect(new PlatformChecker('v1.1.0').isValid).to.be.equal(true);
    expect(new PlatformChecker('v1.2.0').isValid).to.be.equal(true);
  });

  it('should return isValid for Node v0.12', function() {
    expect(new PlatformChecker('v0.12.0').isValid).to.be.equal(true);
    expect(new PlatformChecker('v0.12.15').isValid).to.be.equal(true);
    expect(new PlatformChecker('v0.12.30').isValid).to.be.equal(true);
    expect(new PlatformChecker('v0.10.0').isValid).to.be.equal(false);
  });

  it('should return isValid for Node v0.13', function() {
    expect(new PlatformChecker('v0.13.0').isValid).to.be.equal(true);
    expect(new PlatformChecker('v0.13.15').isValid).to.be.equal(true);
    expect(new PlatformChecker('v0.13.30').isValid).to.be.equal(true);
  });

  it('should return isValid for Node v4', function() {
    expect(new PlatformChecker('v4.0.0').isValid).to.be.equal(true);
    expect(new PlatformChecker('v4.0.15').isValid).to.be.equal(true);
    expect(new PlatformChecker('v4.0.30').isValid).to.be.equal(true);
    expect(new PlatformChecker('v4.1.0').isValid).to.be.equal(true);
    expect(new PlatformChecker('v4.2.0').isValid).to.be.equal(true);
  });

  it('should return isValid for Node v5', function() {
    expect(new PlatformChecker('v5.0.0').isValid).to.be.equal(true);
  });
});
