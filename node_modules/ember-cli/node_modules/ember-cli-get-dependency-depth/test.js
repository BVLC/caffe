var getDependencyDepth = require('./');
var expect = require('chai').expect;

describe('getDependencyDepth', function() {
  it('does what it does', function() {
    expect(getDependencyDepth('')).to.eql('../../..');
    expect(getDependencyDepth('foo')).to.eql('../../..');
    expect(getDependencyDepth('foo/bar')).to.eql('../../../..');
    expect(getDependencyDepth('foo/bar/baz')).to.eql('../../../../..');
  });
});
