'use strict';

var expect = require('chai').expect;
var p = require('../../../preprocessors');
var Registry = require('../../../');

describe('defaultRegistry', function() {
  var fakeApp;
  beforeEach(function() {
    fakeApp = {
      dependencies: function() { }
    };
  });

  it('creates a new Registry instance', function() {
    var registry = p.defaultRegistry(fakeApp);

    expect(registry).to.an.instanceof(Registry);
  });
});
