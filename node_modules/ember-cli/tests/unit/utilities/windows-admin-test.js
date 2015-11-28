'use strict';

var win = require('../../../lib/utilities/windows-admin');
var expect = require('chai').expect;
var MockUI = require('../../helpers/mock-ui');

describe('windows-admin', function() {
  before(function() {
    this.originalPlatform = process.platform;
  });

  after(function () {
    Object.defineProperty(process, 'platform', {
      value: this.originalPlatform
    });
  });

  it('attempts to determine admin rights if Windows', function(done) {
    Object.defineProperty(process, 'platform', {
      value: 'win'
    });

    win.checkWindowsElevation(new MockUI()).then(function (result) {
      expect(result).to.be.ok;
      expect(result.windows).to.be.true;
      done();
    });
  });

  it('does not attempt to determine admin rights if not on Windows', function(done) {
    Object.defineProperty(process, 'platform', {
      value: 'MockOS'
    });

    win.checkWindowsElevation(new MockUI()).then(function (result) {
      expect(result).to.be.ok;
      expect(result.windows).to.be.false;
      done();
    });
  });
});
