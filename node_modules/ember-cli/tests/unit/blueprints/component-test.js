'use strict';

var Blueprint = require('../../../lib/models/blueprint');
var expect    = require('chai').expect;

describe('blueprint - component', function(){
  describe('entityName', function(){
    it('throws error when hyphen is not present', function(){
      var blueprint = Blueprint.lookup('component');

      expect(function() {
        var nonConformantComponentName = 'form';
        blueprint.normalizeEntityName(nonConformantComponentName);
      }).to.throw(/must include a hyphen in the component name/);
    });


    it('keeps existing behavior by calling Blueprint.normalizeEntityName', function(){
      var blueprint = Blueprint.lookup('component');

      expect(function() {
        var nonConformantComponentName = 'x-form/';
        blueprint.normalizeEntityName(nonConformantComponentName);
      }).to.throw(/trailing slash/);
    });
  });
});
