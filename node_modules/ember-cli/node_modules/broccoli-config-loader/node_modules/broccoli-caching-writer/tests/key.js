'use strict';
var chai = require('chai'), expect = chai.expect;
var Key = require('../key');

describe('Key', function() {
  describe('construction', function() {
    it('requires fullPath', function() {
      expect(function() {
        new Key({ });
      }).to.throw(Error, 'entry requires fullPath');

      expect(function() {
        new Key({ fullPath: null });
      }).to.throw(Error, 'entry requires fullPath');

      expect(function() {
        new Key({ fullPath: undefined});
      }).to.throw(Error, 'entry requires fullPath');
    });

    it('requires fullPath + isDirectory', function() {
      expect(function() {
        new Key({ fullPath: 'the/full/path', });
      }).to.throw(Error, 'entry requires isDirectory function');

      expect(function() {
        new Key({ fullPath: 'the/full/path', isDirectory: undefined});
      }).to.throw(Error, 'entry requires isDirectory function');

      expect(function() {
        new Key({ fullPath: 'the/full/path', isDirectory: true});
      }).to.throw(Error, 'entry requires isDirectory function');
    });
  });
});
