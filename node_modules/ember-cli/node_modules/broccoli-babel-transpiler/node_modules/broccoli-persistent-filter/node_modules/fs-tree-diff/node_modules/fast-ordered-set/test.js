/* FastOrderedSet tests */
// source: https://github.com/rquinlivan/set-js/blob/master/test/set_test.js
var expect = require('chai').expect;
var FastOrderedSet = require('./');

describe('FastOrderedSet', function() {

  var obj;

  beforeEach(function() {
    obj = new FastOrderedSet();
  });

  describe('id', function() {
    it('explicit id property', function() {
      obj.add({id:1});
      obj.add({id:2});
      obj.add({id:3});

      expect(obj.size).to.be.eql(3);

      obj.add({id:1});
      obj.add({id:2});
      obj.add({id:3});
      expect(obj.size).to.be.eql(3);

      expect(obj.has({id: 1})).to.be.true;
      expect(obj.has({id: 2})).to.be.true;
      expect(obj.has({id: 3})).to.be.true;
      expect(obj.has({id: 4})).to.be.false;
      expect(obj.has(true)).to.be.false;
      expect(obj.has(false)).to.be.false;
      expect(obj.has(null)).to.be.false;
      expect(obj.has(undefined)).to.be.false;
    });

    it('primitives', function() {
      obj.add(1);
      obj.add(2);
      obj.add(3);

      expect(obj.size).to.be.eql(3);

      obj.add(1);
      obj.add(2);
      obj.add(3);

      expect(obj.size).to.be.eql(3);

      expect(obj.has(1)).to.be.true;
      expect(obj.has(2)).to.be.true;
      expect(obj.has(3)).to.be.true;
      expect(obj.has(4)).to.be.false;
      expect(obj.has(5)).to.be.false;
      expect(obj.has(6)).to.be.false;

      obj.add(null);
      obj.add(null);

      expect(obj.size).to.be.eql(4);
      expect(obj.has(null)).to.be.true;
      expect(obj.has(undefined)).to.be.false;

      obj.add(undefined);
      obj.add(undefined);

      expect(obj.size).to.be.eql(5);
      expect(obj.has(null)).to.be.true;
      expect(obj.has(undefined)).to.be.true;

      obj.delete(undefined);
      obj.delete(undefined);

      expect(obj.size).to.be.eql(4);
      expect(obj.has(null)).to.be.true;
      expect(obj.has(undefined)).to.be.false;

      obj.delete(null);
      obj.delete(null);

      expect(obj.size).to.be.eql(3);
      expect(obj.has(null)).to.be.false;
      expect(obj.has(undefined)).to.be.false;
    });
  });

  describe('#add', function() {
    beforeEach(function() {
      obj.add(1);
    });

    it('can fetch a Number after adding', function() {
      expect(obj.has(1)).to.be.true;
    });

    it("doesn't return a value it didn't add", function() {
      expect(obj.has(2)).to.be.false;
    });

    it('has a size of one', function() {
      expect(obj.size).to.be.eql(1);
    });
  });

  describe('#delete', function() {

    beforeEach(function() {
      obj = new FastOrderedSet([1,2,3,2,2,1,3]);
    });

    describe('when I delete one value', function() {

      beforeEach(function() {
        obj.delete(2);
      });

      it('should contain [1,3]', function() {
        expect(obj.values).to.eql([1,3]);
      });

      it('should have a size of 2', function() {
        expect(obj.size).to.be.eql(2);
      });
    });
  });

  describe('#union', function() {
    var setOne, setTwo;

    describe('union of two overlapping sets', function() {

      var setOne, setTwo, union, union2;

      beforeEach(function () {
        setOne = new FastOrderedSet([1,2,3,4]);
        setTwo = new FastOrderedSet([3,4,5,6]);
        union = setOne.union(setTwo);
        union2 = setTwo.union(setOne);
      });

      it('has the correct number of items', function() {
        expect(union.size).to.be.eql(6);
      });

      it('returns a FastOrderedSet', function() {
        expect(union instanceof FastOrderedSet).to.be.eql(true);
      });

      it('is commutative', function() {
        expect(union.values.sort()).to.be.eql(union2.values.sort());
      });

      describe('when I change the items in setOne', function() {

        beforeEach(function () {
          setOne.delete(1);
          setTwo.delete(6);
        });

        it('should NOT change the size of the union we calculated before', function () {
          expect(union.size).to.be.eql(6);
        });

        describe('and I calculate the union again', function() {
          var unionTwo;

          beforeEach(function() {
            unionTwo = setOne.union(setTwo);
          });

          it('has the correct number of items', function() {
            expect(unionTwo.size).to.be.eql(4);
          });
        });
      });
    });

    describe('using a custom id', function() {
      beforeEach( function() {
        setOne = new FastOrderedSet([{myId: 1, v: 'alpha'}, {myId: 2, v: 'beta'}], 'myId');
        setTwo = new FastOrderedSet([{myId: 1, v: 'charlie'}, {myId: 3, v: 'delta'}], 'myId');
      });

      it('propagates the id', function() {
        var result = setOne.union(setTwo);
        expect(result.size).to.eq(3);
      });
    });
  });

  describe('#intersection', function() {
    var setOne, setTwo, setThree;

    beforeEach(function () {
      setOne = new FastOrderedSet([1,2,3,4]);
      setTwo = new FastOrderedSet([3,4,5,6]);
    });

    it('returns a FastOrderedSet', function() {
      expect(setOne.intersection(setTwo) instanceof FastOrderedSet).to.be.true;
    });

    describe('intersection of two overlapping sets', function() {

      var setOne, setTwo, intersection, intersection2;

      beforeEach(function () {
        setOne = new FastOrderedSet([1,2,3,4]);
        setTwo = new FastOrderedSet([3,4,5,6]);
        intersection = setOne.intersection(setTwo);
        intersection2 = setTwo.intersection(setOne);
      });

      it('has the correct number of items', function() {
        expect(intersection.size).to.be.eql(2);
      });

      it('returns a FastOrderedSet', function() {
        expect(intersection instanceof FastOrderedSet).to.be.true;
      });

      it('is commutative', function() {
        expect(intersection.values).to.be.eql(intersection2.values);
      });

      describe('when I change the items in setOne', function() {

        beforeEach(function() {
          setOne.delete(3);
        });

        it('should NOT change the size of the intersection we calculated before', function() {
          expect(intersection.size).to.be.eql(2);
        });

        describe('and I calculate the intersection again', function() {
          var intTwo;

          beforeEach(function() {
            intTwo = setOne.intersection(setTwo);
          });

          it('has the correct number of items', function() {
            expect(intTwo.size).to.be.eql(1);
          });
        });
      });
    });

    describe('the intersection of two sets with no items in common', function() {
      var setOne, setTwo, intersection;

      beforeEach(function() {
        setOne = new FastOrderedSet([1,2,3,4]);
        setTwo = new FastOrderedSet([5,6,7,8]);
        intersection = setOne.intersection(setTwo);
      });

      it('returns an empty set', function() {
        expect(intersection.size).to.be.eql(0);
      });
    });

    describe('using a custom id', function() {
      beforeEach( function() {
        setOne = new FastOrderedSet([{myId: 1, v: 'alpha'}, {myId: 2, v: 'beta'}], 'myId');
        setTwo = new FastOrderedSet([{myId: 1, v: 'charlie'}, {myId: 3, v: 'delta'}], 'myId');
        setThree = new FastOrderedSet([{myId: 1, v: 'echo'}, {myId: 3, v: 'gamma'}], 'myId');
      });

      it('propagates the id', function() {
        var result = setOne.intersection(setTwo).intersection(setThree);
        expect(result.size).to.eq(1);
      });
    });
  });

  describe('#xor', function () {
    var setOne, setTwo, diff1, diff2;

    beforeEach(function () {
      setOne = new FastOrderedSet([1,2,3]);
      setTwo = new FastOrderedSet([1,2,3,4]);
      diff1 = setOne.xor(setTwo);
      diff2 = setTwo.xor(setOne);
    });

    it('returns a FastOrderedSet', function() {
      expect(diff1 instanceof FastOrderedSet).to.be.true;
    });

    it('returns the xor between the sets', function() {
      expect(diff1.size).to.be.eql(1);
      expect(diff1.has(4)).to.be.true;
    });

    it('is commutative', function() {
      expect(diff1.values).to.be.eql(diff2.values);
    });

    describe('the xor between two identical sets', function() {
      var setThree, setFour, diff3;

      beforeEach(function() {
        setThree = new FastOrderedSet(['a','b','c']);
        setFour = new FastOrderedSet(['b','c','a']);
        diff3 = setThree.xor(setFour);
      });

      it('returns a FastOrderedSet', function() {
        expect(diff3 instanceof FastOrderedSet).to.be.true;
      });

      it('has size() of zero', function() {
        expect(diff3.size).to.be.eql(0);
      });
    });

    describe('using a custom id', function() {
      beforeEach( function() {
        setOne = new FastOrderedSet([{myId: 1, v: 'alpha'}, {myId: 2, v: 'beta'}], 'myId');
        setTwo = new FastOrderedSet([{myId: 1, v: 'charlie'}, {myId: 3, v: 'delta'}], 'myId');
      });

      it('propagates the id', function() {
        var result = setOne.xor(setTwo);
        expect(result.values.map(function (value) {
          return value.v;
        })).to.deep.equal(['delta', 'beta']);
      });
    });
  });

  describe('#difference', function () {
    var setOne, setTwo, diff1, diff2;

    beforeEach(function () {
      setOne = new FastOrderedSet([1,2,3]);
      setTwo = new FastOrderedSet([1,2,3,4]);
      diff1 = setOne.difference(setTwo);
      diff2 = setTwo.difference(setOne);
    });

    it('returns a FastOrderedSet', function() {
      expect(diff1 instanceof FastOrderedSet).to.be.true;
    });

    it('returns the difference between the sets', function() {
      expect(diff1.size).to.be.eql(0);
      expect(diff2.size).to.be.eql(1);
      expect(diff2.values).to.be.eql([4]);
    });

    it('is NOT commutative', function() {
      expect(diff1.values).to.not.be.eql(diff2.values);
    });

    describe('the set difference between two identical sets', function() {
      var setThree, setFour, diff3;

      beforeEach(function() {
        setThree = new FastOrderedSet(['a','b','c']);
        setFour = new FastOrderedSet(['b','c','a']);
        diff3 = setThree.difference(setFour);
      });

      it('returns a FastOrderedSet', function() {
        expect(diff3 instanceof FastOrderedSet).to.be.true;
      });

      it('has size() of zero', function() {
        expect(diff3.size).to.be.eql(0);
      });
    });

    describe('using a custom id', function() {
      beforeEach( function() {
        setOne = new FastOrderedSet([{myId: 1, v: 'alpha'}, {myId: 2, v: 'beta'}], 'myId');
        setTwo = new FastOrderedSet([{myId: 1, v: 'charlie'}, {myId: 3, v: 'delta'}], 'myId');
      });

      it('propagates the id', function() {
        var result = setOne.difference(setTwo);
        expect(result.values.map(function (value) {
          return value.v;
        })).to.deep.equal(['beta']);
      });
    });
  });

  describe('method chaining', function() {
    var setOne, setTwo, retVal;

    beforeEach(function() {
      setOne = new FastOrderedSet([1,2,3,4]);
      setTwo = new FastOrderedSet([2,3,4]);
    });


    describe('#delete', function() {

      beforeEach(function () {
        retVal = setOne.delete(1);
      });

      it('returns to me a reference to the updated set with the item deleted', function() {
        expect(retVal instanceof FastOrderedSet).to.be.true;
        expect(retVal.has(1)).to.be.false;
        expect(retVal.has(2)).to.be.true;
        expect(retVal.has(3)).to.be.true;
        expect(retVal.has(4)).to.be.true;
        expect(retVal.size).to.be.eql(3);
      });
    });

    describe('#add', function() {

      beforeEach(function() {
        retVal = setOne.add(5);
      });

      it('returns to me a reference to the updated set with the item deleted', function() {
        expect(retVal instanceof FastOrderedSet).to.be.true;
        expect(retVal.has(1)).to.be.true;
        expect(retVal.has(2)).to.be.true;
        expect(retVal.has(3)).to.be.true;
        expect(retVal.has(4)).to.be.true;
        expect(retVal.has(5)).to.be.true;
        expect(retVal.size).to.be.eql(5);
      });
    });
  });
});
