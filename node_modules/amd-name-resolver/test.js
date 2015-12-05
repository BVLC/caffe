'use strict';

var expect = require('chai').expect;
var moduleResolve = require('./index');

describe('module resolver', function () {
  it('should resolve relative sibling', function() {
    expect(moduleResolve('./foo', 'example/bar')).to.eql('example/foo');
  });

  it('should resolve relative parent', function() {
    expect(moduleResolve('../foo', 'example/bar/baz')).to.eql('example/foo');
  });

  it('should be a pass through if absolute', function() {
    expect(moduleResolve('foo/bar', 'example/')).to.eql('foo/bar');
  });

  it('should throw parent module of root is accesed', function() {
    expect(function() {
      return moduleResolve('../../bizz', 'example')
    }).to.throw(/Cannot access parent module of root/);
  });
});