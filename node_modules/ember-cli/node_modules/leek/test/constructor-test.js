'use strict';

var assert = require('chai').assert,
    Leek   = require('../lib/leek'),
    ok     = assert.ok,
    equal  = assert.equal,
    throws = assert.throws,
    leek;

describe('constructor', function() {
  it('exists', function() {
    ok(Leek);
  });

  it('asserts if options are not specified', function() {
    throws(function() {
      leek = new Leek();
    }, 'You need to specify the options.');
  });

  it('asserts if tracking code is not specified', function() {
    throws(function() {
      leek = new Leek({
        globalName: 'ember-cli',
        name:       'cli',
        version:    '0.0.23'
      });
    }, 'You need to specify the tracking code.');
  });

  it('should set options', function() {
    leek = new Leek({
      trackingCode: 'xxxxx',
      globalName:   'ember-cli',
      name:         'cli',
      clientId:     'things',
      version:      '0.0.1'
    });

    equal(leek.trackingCode, 'xxxxx', 'tracking code is correct');
    equal(leek.globalName, 'ember-cli', 'name is correct');
    equal(leek.name, 'cli', 'name is correct');
    equal(leek.version, '0.0.1', 'version is correct');
  });

  it('default version is set if it\'s not specified', function() {
    leek = new Leek({
      trackingCode: 'xxxxx',
      globalName:   'ember-cli',
      name:         'cli',
      clientId:     'things'
    });

    equal(leek.version, '', 'version is an empty string');
  });

  it('should have public API methods', function() {
    leek = new Leek({
      trackingCode: 'xxxxx',
      globalName:   'ember-cli',
      name:         'cli'
    });

    ok(leek.track);
    ok(leek.trackError);
    ok(leek.trackTiming);
    ok(leek.trackEvent);
  });

  it('should have private API methods', function() {
    leek = new Leek({
      trackingCode: 'xxxxx',
      globalName:   'ember-cli',
      name:         'cli'
    });

    ok(leek._getConfigObject);
  });
});
