'use strict';

var assert    = require('chai').assert,
    ok        = assert.ok,
    deepEqual = assert.deepEqual,
    rewire    = require('rewire'),
    Leek      = rewire('../lib/leek'),
    called    = false,
    leek,
    params;

describe('track', function() {
  before(function() {
    leek = new Leek({
      trackingCode: 'xxxxx',
      globalName:   'ember-cli',
      name:         'cli',
      clientId:     'things',
      version:      '0.0.1'
    });

    leek._enqueue = function(type, meta) {
      called = true;
      params = {
        type: type,
        meta: meta
      };
    };
  });

  after(function() {
    called = false;
    params = null;
  });

  it('enqueue is called with paramaters', function() {
    leek.track({
      name:    'test',
      message: 'test-test'
    });

    ok(called, 'enqueue was called');
    deepEqual(params, {
      type: 'appview',
      meta: {
        name:    'test',
        message: 'test-test'
      }
    }, 'parameters are correct');
  });
});

describe('trackError', function() {
  before(function() {
    leek = new Leek({
      trackingCode: 'xxxxx',
      globalName:   'ember-cli',
      name:         'cli',
      clientId:     'things',
      version:      '0.0.1'
    });

    leek._enqueue = function(type, meta) {
      called = true;
      params = {
        type: type,
        meta: meta
      };
    };
  });

  after(function() {
    called = false;
    params = null;
  });

  it('enqueue is called with paramaters', function() {
    leek.trackError({
      description: 'ZOMG ERROR',
      isFatal:     true
    });

    ok(called, 'enqueue was called');
    deepEqual(params, {
      type: 'exception',
      meta: {
        description: 'ZOMG ERROR',
        fatal:       true
      }
    }, 'parameters are correct');
  });
});

describe('trackTiming', function() {
  before(function() {
    leek = new Leek({
      trackingCode: 'xxxxx',
      globalName:   'ember-cli',
      name:         'cli',
      clientId:     'things',
      version:      '0.0.1'
    });

    leek._enqueue = function(type, meta) {
      called = true;
      params = {
        type: type,
        meta: meta
      };
    };
  });

  after(function() {
    called = false;
    params = null;
  });

  it('enqueue is called with paramaters', function() {
    leek.trackTiming({
      category: 'blah',
      variable: 'foo',
      label:    'bar',
      value:    '100ms'
    });

    ok(called, 'enqueue was called');
    deepEqual(params, {
      type: 'timing',
      meta: {
        category: 'blah',
        variable: 'foo',
        label:    'bar',
        value:    '100ms'
      }
    }, 'parameters are correct');
  });
});

describe('trackEvent', function() {
  before(function() {
    leek = new Leek({
      trackingCode: 'xxxxx',
      globalName:   'ember-cli',
      name:         'cli',
      clientId:     'things',
      version:      '0.0.1'
    });

    leek._enqueue = function(type, meta) {
      called = true;
      params = {
        type: type,
        meta: meta
      };
    };
  });

  after(function() {
    called = false;
    params = null;
  });

  it('enqueue is called with paramaters', function() {
    leek.trackEvent({
      name:     'test',
      category: 'test-test',
      label:    'test-label',
      value:    'test-value'
    });

    ok(called, 'enqueue was called');
    deepEqual(params, {
      type: 'event',
      meta: {
        name:     'test',
        category: 'test-test',
        label:    'test-label',
        value:    'test-value'
      }
    }, 'parameters are correct');
  });
});