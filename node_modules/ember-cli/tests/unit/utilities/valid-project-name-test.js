'use strict';

var validProjectName = require('../../../lib/utilities/valid-project-name');
var expect             = require('chai').expect;

describe('validate project name', function(){
  it('invalidates nonconformant project name', function(){
    var nonConformantName = 'app';
    var validated = validProjectName(nonConformantName);

    expect(validated).to.not.be.ok;
  });


  it('validates conformant project name', function(){
    var conformantName = 'my-app';
    var validated = validProjectName(conformantName);

    expect(validated).to.be.ok;
  });
});
