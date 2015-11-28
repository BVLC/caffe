'use strict';

var validComponentName = require('../../../lib/utilities/valid-component-name');
var SilentError        = require('silent-error');
var expect             = require('chai').expect;

describe('validate component name', function(){
  it('throws silent error when hyphen is not present', function(){
    var nonConformantName = 'form';

    expect(function() {
      validComponentName(nonConformantName);
    }).to.throw(SilentError, /must include a hyphen in the component name/);
  });


  it('returns the entity name', function(){
    var conformantName = 'x-form';
    var validatedName = validComponentName(conformantName);

    expect(validatedName).to.be.equal(conformantName);
  });
});
