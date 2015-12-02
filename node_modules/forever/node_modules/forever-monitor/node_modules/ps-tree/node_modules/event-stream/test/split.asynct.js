var es = require('event-stream')
  , it = require('it-is').style('colour')
  , d = require('d-utils')
  , join = require('path').join
  , fs = require('fs')
  , Stream = require('stream').Stream

exports ['pipeable'] = function (test) {
  var readme = join(__dirname, 'readme.markdown')
    , expected = fs.readFileSync(readme, 'utf-8').split('\n')
    , cs = es.split()
    , actual = []
    , ended = false

  var a = new Stream ()
  
  a.write = function (l) {
    actual.push(l.trim())
  }
  a.end = function () {

      ended = true
      expected.forEach(function (v,k) {
        it(actual[k]).like(v)
      })
  
      test.done()
    }
  a.writable = true
  
  fs.createReadStream(readme, {flags: 'r'}).pipe(cs)
  cs.pipe(a)  
  
}
