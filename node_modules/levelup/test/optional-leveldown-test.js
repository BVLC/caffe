/* Copyright (c) 2012-2013 LevelUP contributors
 * See list at <https://github.com/rvagg/node-levelup#contributing>
 * MIT +no-false-attribs License <https://github.com/rvagg/node-levelup/blob/master/LICENSE>
 */

var assert  = require('referee').assert
  , refute  = require('referee').refute
  , buster  = require('bustermove')
  , errors  = require('../lib/errors')

function clearCache () {
  delete require.cache[require.resolve('..')]
  delete require.cache[require.resolve('leveldown')]
  delete require.cache[require.resolve('leveldown/package')]
  delete require.cache[require.resolve('../lib/util')]
}

buster.testCase('Optional LevelDOWN', {
    'setUp': clearCache
  , 'tearDown': clearCache

  , 'test getLevelDOWN()': function () {
      var util = require('../lib/util')
      assert.same(util.getLevelDOWN(), require('leveldown'), 'correct leveldown provided')
    }

  , 'test wrong version': function () {
      var levelup = require('..')
      require('leveldown/package').version = '0.0.0'
      assert.exception(levelup.bind(null, '/foo/bar'), function (err) {
        if (err.name != 'LevelUPError')
          return false
        if (!/Installed version of LevelDOWN \(0\.0\.0\) does not match required version \(~\d+\.\d+\.\d+\)/.test(err.message))
          return false
        return true
      })
    }

  , 'test no leveldown/package': function () {
      var levelup = require('..')
      // simulate an exception from a require() that doesn't resolved a package
      Object.defineProperty(require.cache, require.resolve('leveldown/package'), {
        get: function() {
          throw new Error('Wow, this is kind of evil isn\'t it?')
        }
      })
      assert.exception(levelup.bind(null, '/foo/bar'), function (err) {
        if (err.name != 'LevelUPError')
          return false
        if ('Could not locate LevelDOWN, try `npm install leveldown`' != err.message)
          return false
        return true
      })
    }

  , 'test no leveldown': function () {
      var levelup = require('..')
      // simulate an exception from a require() that doesn't resolved a package
      Object.defineProperty(require.cache, require.resolve('leveldown'), {
        get: function() {
          throw new Error('Wow, this is kind of evil isn\'t it?')
        }
      })
      assert.exception(levelup.bind(null, '/foo/bar'), function (err) {
        if (err.name != 'LevelUPError')
          return false
        if ('Could not locate LevelDOWN, try `npm install leveldown`' != err.message)
          return false
        return true
      })
    }
})