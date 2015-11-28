'use strict';

var assert = require('chai').assert,
    ok     = assert.ok,
    equal  = assert.equal,
    called = false,
    params = {},
    expected = null,
    rewire = require('rewire'),
    Leek   = rewire('../lib/leek'),
    leek   = null;

describe('trackTiming()', function() {
  before(function() {
    expected = {
      v:   1,
      t:   'timing',
      aip: 1,
      tid: 'xxxxx',
      cid: 'things',
      an: 'ember-cli',
      av: '0.0.1 OSX Mavericks, node v0.11.12-pre',
      utv: 'test',
      utt: '200ms',
      utl: 'broccoli'
    };

    Leek.__set__('request', function(options) {
      called = true;
      params.url = options.url;
      params.qs = options.qs;
    });

    leek = new Leek({
      trackingCode: 'xxxxx',
      globalName:   'ember-cli',
      name:         'cli',
      clientId:     'things',
      version:      '0.0.1'
    });
  });

  after(function() {
    leek = null;
  });

  it('options passed in are correct', function() {
    leek.trackTiming({
      category: 'rebuild' + Date.now(),
      variable: 'test',
      label:    'broccoli',
      value:    '200ms'
    });

    ok(called);

    equal(params.qs.v,   expected.v);
    equal(params.qs.t,   expected.t);
    equal(params.qs.aip, expected.aip);
    equal(params.qs.tid, expected.tid);
    equal(params.qs.an,  expected.an);

    equal(params.qs.utv, expected.utv);
    equal(params.qs.utt, expected.utt);
    equal(params.qs.utl, expected.utl);
  });
});
