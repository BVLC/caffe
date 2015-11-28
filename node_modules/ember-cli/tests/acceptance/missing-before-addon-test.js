'use strict';

var path  = require('path');
var ember = require('../helpers/ember');
var root  = process.cwd();

describe('Acceptance: missing a before/after addon', function() {
  before(function() {
    process.chdir(path.join(root, 'tests', 'fixtures', 'missing-before-addon'));
  });

  after(function() {
    process.chdir(root);
  });

  it('does not break ember-cli', function() {
    return ember(['help']);
  });
});
