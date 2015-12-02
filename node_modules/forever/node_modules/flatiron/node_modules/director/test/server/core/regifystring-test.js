
var assert = require('assert'),
    vows = require('vows'),
    director = require('../../../lib/director');

var callback = function() {
  return true;
};

var testRoute = function(route, callback) {
  var router = new director.Router();
  router.on(route, callback);

  return function(value) {
    return router.dispatch('on', value);
  };
};

vows.describe('director/core/regifyString').addBatch({

  'When using "/home(.*)"': {
    topic: function() {
      return testRoute('/home(.*)', callback);
    },

    'Should match "/homepage"': function(result) {
      assert.isTrue(result('/homepage'));
    },

    'Should match "/home/page"': function(result) {
      assert.isTrue(result('/home/page'));
    },

    'Should not match "/foo-bar"': function(result) {
      assert.isFalse(result('/foo-bar'));
    }
  },

  'When using "/home.*"': {
    topic: function() {
      return testRoute('/home.*', callback);
    },

    'Should match "/homepage"': function(result) {
      assert.isTrue(result('/homepage'));
    },

    'Should match "/home/page"': function(result) {
      assert.isTrue(result('/home/page'));
    },

    'Should not match "/foo-bar"': function(result) {
      assert.isFalse(result('/foo-bar'));
    }
  },

  'When using "/home(page[0-9])*"': {
    topic: function() {
      return testRoute('/home(page[0-9])*', callback);
    },

    'Should match "/home"': function(result) {
      assert.isTrue(result('/home'));
    },

    'Should match "/homepage0", "/homepage1", etc.': function(result) {
      for (i = 0; i < 10; i++) {
        assert.isTrue(result('/homepage' + i));
      }
    },

    'Should not match "/home_page"': function(result) {
      assert.isFalse(result('/home_page'));
    },

    'Should not match "/home/page"': function(result) {
      assert.isFalse(result('/home/page'));
    }
  },

  'When using "/home*"': {
    topic: function() {
      return testRoute('/home*', callback);
    },

    'Should match "/homepage"': function(result) {
      assert.isTrue(result('/homepage'));
    },

    'Should match "/home_page"': function(result) {
      assert.isTrue(result('/home_page'));
    },

    'Should match "/home-page"': function(result) {
      assert.isTrue(result('/home-page'));
    },

    'Should not match "/home/page"': function(result) {
      assert.isFalse(result('/home/page'));
    }
  },
  'When using "/folder/::home"': {
    topic: function() {
      return testRoute('/folder/::home', callback);
    },
    'Should match "/folder/:home"': function(result) {
      assert.isTrue(result('/folder/:home'));
    },
    'Should not match "/folder/::home"': function(result) {
      assert.isFalse(result('/folder/::home'));
    },
    'Should not match "/folder/abc" (the catchall regexp)': function(result) {
      assert.isFalse(result('/folder/abc'));
    }
  }

}).export(module);
