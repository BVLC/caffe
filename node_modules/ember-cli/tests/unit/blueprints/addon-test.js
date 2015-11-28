'use strict';

var Blueprint   = require('../../../lib/models/blueprint');
var MockProject = require('../../helpers/mock-project');
var expect      = require('chai').expect;

describe('blueprint - addon', function(){
  describe('entityName', function(){
    var mockProject;

    beforeEach(function() {
      mockProject = new MockProject();
      mockProject.isEmberCLIProject = function() { return true; };
    });

    afterEach(function() {
      mockProject = null;
    });

    it('throws error when current project is an existing ember-cli project', function(){
      var blueprint = Blueprint.lookup('addon');

      blueprint.project = mockProject;

      expect(function() {
        blueprint.normalizeEntityName('foo');
      }).to.throw('Generating an addon in an existing ember-cli project is not supported.');
    });

    it('works when current project is an existing ember-cli addon', function(){
      mockProject.isEmberCLIAddon = function() { return true; };
      var blueprint = Blueprint.lookup('addon');

      blueprint.project = mockProject;

      expect(function() {
        blueprint.normalizeEntityName('foo');
      }).not.to.throw('Generating an addon in an existing ember-cli project is not supported.');
    });

    it('keeps existing behavior by calling Blueprint.normalizeEntityName', function(){
      var blueprint = Blueprint.lookup('addon');

      blueprint.project = mockProject;

      expect(function() {
        var nonConformantComponentName = 'foo/';
        blueprint.normalizeEntityName(nonConformantComponentName);
      }).to.throw(/trailing slash/);
    });
  });
});
