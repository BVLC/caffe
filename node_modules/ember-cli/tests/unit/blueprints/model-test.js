'use strict';

var Blueprint = require('../../../lib/models/blueprint');
var expect    = require('chai').expect;

var EOL = require('os').EOL;

function makeOpts(opts){
  return {entity: {options: opts}};
}

describe('blueprint - model', function(){
  describe('entityName', function(){
    it('generates individual un-typed properties', function(){
      var blueprint = Blueprint.lookup('model');

      var out = blueprint.locals(makeOpts({
        name: ''
      }));
      console.log(out);
      expect(out.attrs).to.equal('name: DS.attr()');
      expect(out.needs).to.equal('  needs: []');
    });

    it('generates individual typed properties', function(){
      var blueprint = Blueprint.lookup('model');

      var out = blueprint.locals(makeOpts({
        name: 'type'
      }));
      expect(out.attrs).to.equal('name: DS.attr(\'type\')');
      expect(out.needs).to.equal('  needs: []');
    });

    it('accepts camel-cased properties', function(){
      var blueprint = Blueprint.lookup('model');

      var out = blueprint.locals(makeOpts({
        aliceTheCamel: 'humps'
      }));
      console.log(out);
      expect(out.attrs).to.equal('aliceTheCamel: DS.attr(\'humps\')');
      expect(out.needs).to.equal('  needs: []');
    });

    it('accepts dasheriezed properties', function(){
      var blueprint = Blueprint.lookup('model');

      var out = blueprint.locals(makeOpts({
        'dash-to-the-future': 'time-machine'
      }));
      expect(out.attrs).to.equal('dashToTheFuture: DS.attr(\'time-machine\')');
      expect(out.needs).to.equal('  needs: []');
    });

    it('accepts underscored properties', function(){
      var blueprint = Blueprint.lookup('model');

      var out = blueprint.locals(makeOpts({
        /* jshint camelcase: false */
        six_feet_underscored: ''
        /* jshint camelcase: true */
      }));
      console.log(out);
      expect(out.attrs).to.equal('sixFeetUnderscored: DS.attr()');
      expect(out.needs).to.equal('  needs: []');
    });

    it('generates multiple properties', function(){
      var blueprint = Blueprint.lookup('model');

      var out = blueprint.locals(makeOpts({
        one: 'Fish',
        two: 'Fish'
      }));
      console.log(out);
      expect(out.attrs).to.equal('one: DS.attr(\'fish\'),' + EOL + '  two: DS.attr(\'fish\')');
      expect(out.needs).to.equal('  needs: []');
    });

    it('links model by name if type is undefined', function(){
      var blueprint = Blueprint.lookup('model');

      var out = blueprint.locals(makeOpts({
        virtue: 'belongs-to',
        vices: 'has-many'
      }));
      console.log(out);
      expect(out.attrs).to.equal('virtue: DS.belongsTo(\'virtue\'),' + EOL + '  vices: DS.hasMany(\'vice\')');
      expect(out.needs).to.equal('  needs: [\'model:virtue\', \'model:vice\']');
    });

    it('links supplied model name (singularized) if defined', function(){
      var blueprint = Blueprint.lookup('model');

      var out = blueprint.locals(makeOpts({
        tomster: 'belongs-to:leah',
        irlTom: 'has-many:twitterRants'
      }));
      console.log(out);
      expect(out.attrs).to.equal('tomster: DS.belongsTo(\'leah\'),' + EOL + '  irlToms: DS.hasMany(\'twitter-rant\')');
      expect(out.needs).to.equal('  needs: [\'model:leah\', \'model:twitter-rant\']');
    });

  });
});
