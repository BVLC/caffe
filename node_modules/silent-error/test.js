'use strict';

var SilentError = require('./');
var expect      = require('chai').expect;

describe('SilentError', function() {
  var error;

  it('should suppress the stack trace by default', function() {
    error = new SilentError();
    expect(error.suppressStacktrace, 'suppressesStacktrace should be true');
  });

  describe('with EMBER_VERBOSE_ERRORS set', function() {
    beforeEach(function() {
      delete process.env.EMBER_VERBOSE_ERRORS;
    });

    it('should suppress stack when true', function() {
      process.env.EMBER_VERBOSE_ERRORS = 'true';
      error = new SilentError();
      expect(!error.suppressStacktrace, 'suppressesStacktrace should be false');
    });

    it('shouldn\'t suppress stack when false', function() {
      process.env.EMBER_VERBOSE_ERRORS = 'false';
      error = new SilentError();
      expect(error.suppressStacktrace, 'suppressesStacktrace should be true');
    });
  });

  describe('debugOrThrow', function() {
    it('throws non SilentError', function() {
      expect(function() {
        SilentError.debugOrThrow('label', new Error('I AM ERROR'));
      }).to.throw('I AM ERROR');
    });

    it('throws false|null|undefined', function() {
      expect(function() { SilentError.debugOrThrow('label', false);     }).to.throw(false);
      expect(function() { SilentError.debugOrThrow('label', true);      }).to.throw(true);
      expect(function() { SilentError.debugOrThrow('label', undefined); }).to.throw(undefined);
      expect(function() { SilentError.debugOrThrow('label', null);      }).to.throw(null);
    });

    it('doesnt throw with SilentError', function() {
      expect(function() { SilentError.debugOrThrow('label', new SilentError('ERROR')); }).to.not.throw();
    });
  });
});
