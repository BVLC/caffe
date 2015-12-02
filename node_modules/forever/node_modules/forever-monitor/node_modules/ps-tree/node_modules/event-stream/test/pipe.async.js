var es = require('event-stream')
  , it = require('it-is').style('colour')
  , d = require('d-utils')

function makeExamplePipe() {

  return es.pipe(
    es.map(function (data, callback) {
      callback(null, data * 2)
    }),
    es.map(function (data, callback) {
      d.delay(callback)(null, data)    
    }),
    es.map(function (data, callback) {
      callback(null, data + 2)
    }))
}

exports['simple pipe'] = function (test) {

  var pipe = makeExamplePipe()

  pipe.on('data', function (data) {
    it(data).equal(18)
    test.done()
  })
  
  pipe.write(8)

}

exports['read array then map'] = function (test) {

  var readThis = d.map(3, 6, 100, d.id) //array of multiples of 3 < 100
    , first = es.readArray(readThis)
    , read = []
    , pipe =
  es.pipe(
    first,
    es.map(function (data, callback) {
      callback(null, {data: data})      
    }),
    es.map(function (data, callback) {
      callback(null, {data: data})
    }),
    es.writeArray(function (err, array) {
      it(array).deepEqual(d.map(readThis, function (data) {
        return {data: {data: data}}
      }))
      test.done()  
    })
  )

}