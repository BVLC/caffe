'use strict';

var SilentError    = require('silent-error');
var SilentErrorLib = require('../../../lib/errors/silent');
var expect         = require('chai').expect;

describe('SilentError', function() {
  it('return silent-error and print a deprecation', function() {
    expect(SilentErrorLib, 'returns silent-error').to.equal(SilentError);
  });
});
