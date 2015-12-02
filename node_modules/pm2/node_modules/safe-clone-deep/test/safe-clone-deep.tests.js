'use strict';

var expect = require('chai').expect;
var clone = require('../src');

describe('safeCloneDeep', function(){
  var input, output;

  beforeEach(function(){
    var a = {};
    a.a = a;
    a.b = {};
    a.b.a = a;
    a.b.b = a.b;
    a.c = {};
    a.c.b = a.b;
    a.c.c = a.c;
    a.x = 1;
    a.b.x = 2;
    a.c.x = 3;
    a.d = [0,a,1,a.b,2,a.c,3];
    input = a;
  });

  describe('irregular objects', function(){
    it('will clone a Date', function(){
      var a = new Date();
      var b = clone(a);
      expect(a.valueOf()).to.equal(b.valueOf());
      expect(a).to.not.equal(b);
    });

    it ('will clone a Buffer', function(){
      var a = new Buffer('this is a test');
      var b = clone(a);
      expect(a.toString()).to.equal(b.toString());
      expect(a).to.not.equal(b);
    });

    it ('will clone an Error\'s properties', function(){
      var a = new Error("this is a test");
      var b = clone(a);

      expect(a).to.not.equal(b);
      expect(b).to.have.property('name',a.name);
      expect(b).to.have.property('message',a.message);
      expect(b).to.have.property('stack',a.stack);
    });

    it('will not clone an inherited property', function(){
      function Base(){
        this.base = true; 
      }
      function Child(){
        this.child = true;
      }
      Child.prototype = new Base();

      var z = clone(new Child());
      expect(z).to.have.property('child',true);
      expect(z).to.not.have.property('base');
    });
  });

  describe('default circularValue of undefined', function(){
    beforeEach(function(){
      output = clone(input);
    });

    it('will return the expected values on base object', function(){
      expect(input).to.have.property('a',undefined);
      expect(input).to.have.property('b');
      expect(input).to.have.property('x',1);
      expect(input).to.have.property('c');
    });

    it('will return the expected values on nested property', function(){
      expect(input.b).to.exist;
      expect(input.b).to.have.property('a',undefined);
      expect(input.b).to.have.property('b',undefined);
      expect(input.b).to.have.property('x',2);
    });

    it('will return the expected values on secondary nested property', function(){
      expect(input.c).to.exist;
      expect(input.c).to.not.have.property('a');
      expect(input.c).to.have.property('b');
      expect(input.c).to.have.property('c',undefined);
      expect(input.c.b).to.equal(input.b);
      expect(input.c).to.have.property('x',3);
    });
  });
});
