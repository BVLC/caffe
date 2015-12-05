var assert = require('chai').assert
var process = require('./index')

test('it executes command and calls complete', function(done){
  process('echo hello')
    .complete(function(err, stdout){
      assert.equal(stdout, 'hello\n')
      done()
    })
})

test('it calls bad if exit code is bad', function(done){
  process('blarg')
    .good(assert.fail)
    .bad(function(err, stdout, stderr){
      assert(err instanceof Error)
      assert.match(err.message, /blarg: command not found/)
      assert.match(stderr, /blarg: command not found/)
      assert.equal(stdout, '')
      done()
    })
})

test('looks for good pattern', function(done){
  var goodCalled = false
  process('echo hello')
    .goodIfMatches(/hello/)
    .good(function(){
      goodCalled = true
    })
    .complete(function(){
      assert(goodCalled)
      done()
    })
})

test('good pattern can also be string (they get matche literally) - match', function(done){
  var goodCalled = false
  process('echo hello')
    .goodIfMatches('hello')
    .good(function(){
      goodCalled = true
    })
    .complete(function(){
      assert(goodCalled)
      done()
    })
})

test('good pattern can also be string (they get matche literally) - no match', function(done){
  var goodCalled = false
  process('echo hello')
    .goodIfMatches('he.lo')
    .good(function(){
      goodCalled = true
    })
    .complete(function(){
      assert.isFalse(goodCalled)
      done()
    })
})

test('doesnt call good or bad if doesnt find good pattern', function(done){
  process('echo blah')
    .goodIfMatches(/hello/)
    .good(assert.fail)
    .bad(assert.fail)
    .complete(function(){
      done()
    })
})

test('calls bad if times out w/o finding good pattern', function(done){
  process('sleep 3')
    .goodIfMatches(/hello/, 100)
    .good(assert.fail)
    .bad(function(){
      done()
    })
})

test('matches bad pattern', function(done){
  var badCalled = false
  process('echo bad')
    .badIfMatches(/bad/)
    .bad(function(){
      badCalled = true
    })
    .complete(function(){
      assert(badCalled)
      done()
    })
})

test('doesnt call good or bad if doent find bad pattern', function(done){
  process('echo good')
    .badIfMatches(/bad/)
    .bad(assert.fail)
    .good(assert.fail)
    .complete(function(){
      done()
    })
})

test('calls good if times out w/o finding bad pattern', function(done){
  process('sleep 3')
    .badIfMatches(/bad/, 100)
    .bad(assert.fail)
    .good(function(stdout){
      assert.equal(stdout, '')
      done()
    })
})

test('it uses spawn if you give 2 arguments (exe, args)', function(done){
  process('echo', ['good'])
    .complete(function(err, stdout){
      assert.equal(stdout, 'good\n')
      done()
    })
})

test('it kills process', function(done){
  process('sleep 3')
    .kill(function(){
      done()
    })
})

test('it passes stderr and stdout to bad', function(done){
  process('node', ['test_prog.js'])
    .badIfMatches(/hello/)
    .bad(function(err, stdout, stderr){
      assert.equal(stdout, 'hello to stdout\n')
      assert.equal(stderr, 'hello to stderr\n')
      done()
    })
})

test('it passes err to complete', function(done){
  process('node', ['test_prog.js'])
    .badIfMatches(/hello/)
    .complete(function(err, stdout, stderr){
      assert.equal(err.message, 'Found bad match(/hello/): hello to stdout\n')
      done()
    })
})

test('it takes options (exec)', function(done){
  process('pwd')
    .options({cwd: '..'})
    .complete(function(err, stdout){
      var parentDir = stdout.trim().match(/([a-z_]+)$/)[1]
      assert.notEqual(parentDir, 'did_it_work')
      done()
    })
})

test('it takes options (spawn)', function(done){
  process('pwd', ['-L'])
    .options({cwd: '..'})
    .complete(function(err, stdout, stderr){
      var parentDir = stdout.trim().match(/([a-z_]+)$/)[1]
      assert.notEqual(parentDir, 'did_it_work')
      done()
    })
})