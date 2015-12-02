/* Copyright (c) 2012-2013 LevelUP contributors
 * See list at <https://github.com/rvagg/node-levelup#contributing>
 * MIT +no-false-attribs License <https://github.com/rvagg/node-levelup/blob/master/LICENSE>
 */

var common  = require('./common')

  , assert  = require('referee').assert
  , refute  = require('referee').refute
  , buster  = require('bustermove')

buster.testCase('Key and Value Streams', {
    'setUp': function (done) {
      common.commonSetUp.call(this, function () {
        this.dataSpy    = this.spy()
        this.endSpy     = this.spy()
        this.sourceData = []

        for (var i = 0; i < 100; i++) {
          var k = (i < 10 ? '0' : '') + i
          this.sourceData.push({
              type  : 'put'
            , key   : k
            , value : Math.random()
          })
        }

        this.sourceKeys = Object.keys(this.sourceData)
          .map(function (k) { return this.sourceData[k].key }.bind(this))
        this.sourceValues = Object.keys(this.sourceData)
          .map(function (k) { return this.sourceData[k].value }.bind(this))

        this.verify = function (rs, data, done) {
          assert.equals(this.endSpy.callCount, 1, 'Stream emitted single "end" event')
          assert.equals(this.dataSpy.callCount, data.length, 'Stream emitted correct number of "data" events')
          data.forEach(function (d, i) {
            var call = this.dataSpy.getCall(i)
            if (call) {
              //console.log('call', i, ':', call.args[0].key, '=', call.args[0].value, '(expected', d.key, '=', d.value, ')')
              assert.equals(call.args.length, 1, 'Stream "data" event #' + i + ' fired with 1 argument')
              assert.equals(+call.args[0].toString(), +d, 'Stream correct "data" event #' + i + ': ' + d)
            }
          }.bind(this))
          done()
        }.bind(this)

        done()
      }.bind(this))
    }

  , 'tearDown': common.commonTearDown

  , 'test .keyStream()': function (done) {
      this.openTestDatabase(function (db) {
        // execute
        db.batch(this.sourceData.slice(), function (err) {
          refute(err)

          var rs = db.keyStream()
          rs.on('data', this.dataSpy)
          rs.on('end', this.endSpy)
          rs.on('close', this.verify.bind(this, rs, this.sourceKeys, done))
        }.bind(this))
      }.bind(this))
    }

  , 'test .readStream({keys:true,values:false})': function (done) {
      this.openTestDatabase(function (db) {
        // execute
        db.batch(this.sourceData.slice(), function (err) {
          refute(err)

          var rs = db.readStream({ keys: true, values: false })
          rs.on('data', this.dataSpy)
          rs.on('end', this.endSpy)
          rs.on('close', this.verify.bind(this, rs, this.sourceKeys, done))
        }.bind(this))
      }.bind(this))
    }

  , 'test .valueStream()': function (done) {
      this.openTestDatabase(function (db) {
        // execute
        db.batch(this.sourceData.slice(), function (err) {
          refute(err)

          var rs = db.valueStream()
          rs.on('data', this.dataSpy)
          rs.on('end', this.endSpy)
          rs.on('close', this.verify.bind(this, rs, this.sourceValues, done))
        }.bind(this))
      }.bind(this))
    }

  , 'test .readStream({keys:false,values:true})': function (done) {
      this.openTestDatabase(function (db) {
        // execute
        db.batch(this.sourceData.slice(), function (err) {
          refute(err)

          var rs = db.readStream({ keys: false, values: true })
          rs.on('data', this.dataSpy)
          rs.on('end', this.endSpy)
          rs.on('close', this.verify.bind(this, rs, this.sourceValues, done))
        }.bind(this))
      }.bind(this))
    }
})
