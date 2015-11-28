var test = require('tape')
var RSVP = require('rsvp')
var mapSeries = require('./index')



test('mapSeries', function (t) {
  t.test('iterator is called in sequence for each item', function (t) {
    t.plan(6)
    var seq = 0
    mapSeries([0, 1], function (item) {
        t.equal(seq, item)
        return new RSVP.Promise(function (resolve, reject) {
          setTimeout(function () {
            t.equal(seq++, item)
            resolve(item === 0 ? 'foo' : 'bar')
          }, 10)
        })
      })
      .then(function (results) {
        t.equal(seq, 2)
        t.deepEqual(results, ['foo', 'bar'])
      })
  })

  t.test('is rejected on first rejection', function (t) {
    t.plan(2)
    var errorObject = new Error('rejected')
    mapSeries([0, 1], function (item) {
        t.pass('is called once')
        throw errorObject
      })
      .then(function () {
        t.fail('promise should be rejected')
      }, function (err) {
        t.equal(err, errorObject)
      })
  })

  t.test('passes index and array argument to iterator', function (t) {
    t.plan(5)
    var arr = [42, 43]
    mapSeries(arr, function (item, index, array) {
      t.equal(item, index + 42)
      t.equal(array, arr)
    }).then(function (results) {
      t.deepEqual(results, [undefined, undefined])
    })
  })

  t.test('accepts optional thisArg argument', function (t) {
    t.plan(2)
    var obj = {}
    mapSeries([0], function (item) {
      t.equal(this, global)
    })
    mapSeries([0], function (item) {
      t.equal(this, obj)
    }, obj)
  })
})
