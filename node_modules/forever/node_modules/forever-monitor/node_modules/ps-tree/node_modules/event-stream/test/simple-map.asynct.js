
var es = require('event-stream')
  , it = require('it-is')

function writeArray(array, stream) {

  array.forEach( function (j) {
    stream.write(j)
  })
  stream.end()

}

function readStream(stream, done) {

  var array = [] 
  stream.on('data', function (data) {
    array.push(data)
  })
  stream.on('error', done)
  stream.on('end', function (data) {
    done(null, array)
  })

} 

exports ['simple map applied to a stream'] = function (test) {

  var input = [1,2,3,7,5,3,1,9,0,2,4,6]
  //create event stream from

  var doubler = es.map(function (data, cb) {
    cb(null, data * 2)
  })
  
  readStream(doubler, function (err, output) {
    it(output).deepEqual(input.map(function (j) {
      return j * 2
    }))
    test.done()
  })
  
  writeArray(input, doubler)
  
}

exports['pipe two maps together'] = function (test) {

  var input = [1,2,3,7,5,3,1,9,0,2,4,6]
  //create event stream from
  function dd (data, cb) {
    cb(null, data * 2)
  }
  var doubler1 = es.map(dd), doubler2 = es.map(dd)

  doubler1.pipe(doubler2)
  
  readStream(doubler2, function (err, output) {
    it(output).deepEqual(input.map(function (j) {
      return j * 4
    }))
    test.done()
  })
  
  writeArray(input, doubler1)

}

//next:
//
// test pause, resume and drian.
//

// then make a pipe joiner:
//
// plumber (evStr1, evStr2, evStr3, evStr4, evStr5)
//
// will return a single stream that write goes to the first 

exports ['map will not call end until the callback'] = function (test) {

  var ticker = es.map(function (data, cb) {
    process.nextTick(function () {
      cb(null, data * 2)
    })
  })
  ticker.write('x')

  ticker.end()
  ticker.end()
  ticker.end()

  ticker.on('end', function () {
    test.done()
  })
}