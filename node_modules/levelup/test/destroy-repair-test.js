/* Copyright (c) 2012-2013 LevelUP contributors
 * See list at <https://github.com/rvagg/node-levelup#contributing>
 * MIT +no-false-attribs License <https://github.com/rvagg/node-levelup/blob/master/LICENSE>
 */

var levelup   = require('../lib/levelup.js')
  , leveldown = require('leveldown')

  , assert    = require('referee').assert
  , refute    = require('referee').refute
  , buster    = require('bustermove')

buster.testCase('Destroy & Repair', {
    'destroy() is alias for leveldown.destroy()': function () {
      var ldmock = this.mock(leveldown)
        , expect = ldmock.expects('destroy').once()
      levelup.destroy()
      ldmock.verify()
      assert.same(expect.getCall(0).args[0], undefined)
    }

  , 'repair() is alias for leveldown.repair()': function () {
      var ldmock = this.mock(leveldown)
        , expect = ldmock.expects('repair').once()
      levelup.repair()
      ldmock.verify()
      assert.same(expect.getCall(0).args[0], undefined)
    }

  , 'destroy() passes on arguments': function () {
      var ldmock = this.mock(leveldown)
        , args   = [ 'location', function () { } ]
        , expect = ldmock
            .expects('destroy')
            .once()
            .withExactArgs(args[0], args[1])

      levelup.destroy.apply(null, args)
      ldmock.verify()
    }

  , 'repair() passes on arguments': function () {
      var ldmock = this.mock(leveldown)
        , args   = [ 'location', function () { } ]
        , expect = ldmock
            .expects('repair')
            .once()
            .withExactArgs(args[0], args[1])

      levelup.repair.apply(null, args)
      ldmock.verify()
    }

  , 'destroy() substitutes missing callback argument': function () {
      var ldmock = this.mock(leveldown)
        , args   = [ 'location' ]
        , expect = ldmock
            .expects('destroy')
            .once()
            .withArgs(args[0])

      levelup.destroy.apply(null, args)
      ldmock.verify()
      assert.equals(2, expect.getCall(0).args.length)
      assert.isFunction(expect.getCall(0).args[1])
    }

  , 'repair() substitutes missing callback argument': function () {
      var ldmock = this.mock(leveldown)
        , args   = [ 'location' ]
        , expect = ldmock
            .expects('repair')
            .once()
            .withArgs(args[0])

      levelup.repair.apply(null, args)
      ldmock.verify()
      assert.equals(2, expect.getCall(0).args.length)
      assert.isFunction(expect.getCall(0).args[1])
    }
})