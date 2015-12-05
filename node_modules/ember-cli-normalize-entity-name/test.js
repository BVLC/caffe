'use strict';

var expect = require('chai').expect;
var SilentError = require('silent-error');
var normalizeEntityName = require('./');

describe('normalizeEntityName', function() {
  it('throws if no entity name is provides', function() {
    expect(function() {
      normalizeEntityName();
    }).to.throw(SilentError, 'SilentError: The `ember generate <entity-name>` command requires an entity name to be specified. For more details, use `ember help`.');

    expect(function() {
      normalizeEntityName('');
    }).to.throw(SilentError, 'SilentError: The `ember generate <entity-name>` command requires an entity name to be specified. For more details, use `ember help`.');

    expect(function() {
      normalizeEntityName(undefined);
    }).to.throw(SilentError, 'SilentError: The `ember generate <entity-name>` command requires an entity name to be specified. For more details, use `ember help`.');

    expect(function() {
      normalizeEntityName(null);
    }).to.throw(SilentError, 'SilentError: The `ember generate <entity-name>` command requires an entity name to be specified. For more details, use `ember help`.');
  });

  it('throws with trailing slash', function() {
    expect(function() {
      normalizeEntityName('asdf/');
    }).to.throw(SilentError, 'SilentError: You specified "asdf/", but you can\'t use a trailing slash as an entity name with generators. Please re-run the command with "asdf".');
  });

  it('acts like an identity function if the input was valid', function() {
      expect(normalizeEntityName('asdf')).to.eql('asdf');
  });
});
