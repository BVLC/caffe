var EventEmitter = require('events').EventEmitter
var childProcess = require('child_process')
var deploy = require('../deploy.js')

describe('deploy', function() {
  describe('deployForEnv', function() {

    var spawnCalls
    var spawnNotifier
    beforeEach(function() {
      spawnCalls = []
      spawnNotifier = new EventEmitter()
    })

    childProcess.spawn = function(cmd, args, options) {
      spawnCalls.push(arguments)
      var proc = new EventEmitter()

      process.nextTick(function() {
        spawnNotifier.emit('spawned', proc)
      })

      return proc
    }

    var conf
    beforeEach(function() {
      conf = {
        staging: {
          user: 'user',
          host: 'host',
          repo: 'repo',
          path: 'path',
          ref: 'ref'
        }
      }
    })

    it('is a function', function() {
      deploy.deployForEnv.should.be.a.Function
    })

    it('returns false', function() {
      var ret = deploy.deployForEnv(conf, 'staging', [], function() {})
      ret.should.be.false
    })

    describe('deploy_conf validation', function() {
      it('requires user', function(done) {
        delete conf.staging.user
        deploy.deployForEnv(conf, 'staging', [], function(err, args) {
          err.should.be.an.Object
          err.code.should.equal(302)
          err.message.should.match('Missing required property: user')
          done()
        })
      })

      it('requires host', function(done) {
        delete conf.staging.host
        deploy.deployForEnv(conf, 'staging', [], function(err, args) {
          err.should.be.an.Object
          err.code.should.equal(302)
          err.message.should.match('Missing required property: host')
          done()
        })
      })

      it('requires repo', function(done) {
        delete conf.staging.repo
        deploy.deployForEnv(conf, 'staging', [], function(err, args) {
          err.should.be.an.Object
          err.code.should.equal(302)
          err.message.should.match('Missing required property: repo')
          done()
        })
      })

      it('requires path', function(done) {
        delete conf.staging.path
        deploy.deployForEnv(conf, 'staging', [], function(err, args) {
          err.should.be.an.Object
          err.code.should.equal(302)
          err.message.should.match('Missing required property: path')
          done()
        })
      })

      it('requires ref', function(done) {
        delete conf.staging.ref
        deploy.deployForEnv(conf, 'staging', [], function(err, args) {
          err.should.be.an.Object
          err.code.should.equal(302)
          err.message.should.match('Missing required property: ref')
          done()
        })
      })
    })

    describe('spawning child processes', function() {
      context('successfully', function() {
        it('invokes our callback with the supplied arguments', function(done) {
          var argsIn = [1,2,'three','four']
          spawnNotifier.on('spawned', function(proc) {
            proc.emit('close', 0)
          })
          deploy.deployForEnv(conf, 'staging', argsIn, function(err, argsOut) {
            argsOut.should.eql(argsIn)
            done()
          })
        })

        it('invokes sh -c', function(done) {
          spawnNotifier.on('spawned', function(proc) {
            proc.emit('close', 0)
          })
          deploy.deployForEnv(conf, 'staging', [], function(err, args) {
            spawnCalls.length.should.equal(1)
            spawnCalls[0][0].should.equal('sh')
            spawnCalls[0][1].should.be.an.Array
            spawnCalls[0][1][0].should.equal('-c')
            done()
          })
        })

        it('echoes a JSON blob', function(done) {
          spawnNotifier.on('spawned', function(proc) {
            proc.emit('close', 0)
          })
          deploy.deployForEnv(conf, 'staging', [], function(err, args) {
            spawnCalls.length.should.equal(1)
            spawnCalls[0][1][1].should.be.a.String

            var pipeFrom = spawnCalls[0][1][1].split(/\s*\|\s*/)[0]
            pipeFrom.should.be.ok

            var echoJSON = pipeFrom.match(/^echo '(.+?)'/)[1]
            echoJSON.should.be.ok

            var echoData = JSON.parse(echoJSON)
            echoData.should.be.an.Object
            echoData.should.eql(conf.staging)
            done()
          })
        })

        it('pipes to deploy', function(done) {
          spawnNotifier.on('spawned', function(proc) {
            proc.emit('close', 0)
          })
          deploy.deployForEnv(conf, 'staging', [], function(err, args) {
            spawnCalls.length.should.equal(1)
            spawnCalls[0][1][1].should.be.a.String
            var pipeTo = spawnCalls[0][1][1].split(/\s*\|\s*/)[1]
            pipeTo.should.be.ok
            pipeTo.should.match(/\/deploy\s*$/)
            done()
          })
        })
      })

      context('with errors', function() {
        it('calls back with the error stack, if present', function(done) {
          var error = { stack: 'this is my stack'}
          spawnNotifier.on('spawned', function(proc) {
            proc.emit('error', error)
          })
          deploy.deployForEnv(conf, 'staging', [], function(err, args) {
            err.should.be.a.String
            err.should.eql(error.stack)
            done()
          })
        })

        it('calls back with the error object, if no stack is present', function(done) {
          var error = { abc: 123 }
          spawnNotifier.on('spawned', function(proc) {
            proc.emit('error', error)
          })
          deploy.deployForEnv(conf, 'staging', [], function(err, args) {
            err.should.be.an.Object
            err.should.eql(error)
            done()
          })
        })
      })

      context('for multiple hosts', function() {
        var hosts = ['1.1.1.1', '2.2.2.2', '3.3.3.3', '4.4.4.4']

        beforeEach(function() {
          conf.staging.host = hosts
        })

        it('runs each host in series', function(done) {
          var spawnCount = 0
          spawnNotifier.on('spawned', function(proc) {
            spawnCount += 1
            spawnCount.should.equal(1)
            process.nextTick(function() {
              proc.emit('close', 0)
              spawnCount -= 1
            })
          })
          deploy.deployForEnv(conf, 'staging', [], function(err, args) {
            done()
          })
        })

        it('echoes JSON blobs with customized host attributes', function(done) {
          var spawnCount = 0

          spawnNotifier.on('spawned', function(proc) {
            var pipeFrom = spawnCalls[spawnCount][1][1].split(/\s*\|\s*/)[0]
            pipeFrom.should.be.ok

            var echoJSON = pipeFrom.match(/^echo '(.+?)'/)[1]
            echoJSON.should.be.ok

            var echoData = JSON.parse(echoJSON)
            echoData.should.be.an.Object

            echoData.ref.should.eql(conf.staging.ref)
            echoData.user.should.eql(conf.staging.user)
            echoData.repo.should.eql(conf.staging.repo)
            echoData.path.should.eql(conf.staging.path)
            echoData.host.should.eql(hosts[spawnCount])

            spawnCount += 1

            process.nextTick(function() {
              proc.emit('close', 0)
            })
          })

          deploy.deployForEnv(conf, 'staging', [], function(err, args) {
            spawnCount.should.eql(4)
            done()
          })
        })

        it('invokes our callback with supplied argument arrays', function(done) {
          var argsIn = [1,2,'three','four']
          spawnNotifier.on('spawned', function(proc) {
            proc.emit('close', 0)
          })

          deploy.deployForEnv(conf, 'staging', argsIn, function(err, argsOut) {
            argsOut.should.be.an.Array
            argsOut.length.should.eql(4)
            argsOut[0].should.eql(argsIn)
            argsOut[1].should.eql(argsIn)
            argsOut[2].should.eql(argsIn)
            argsOut[3].should.eql(argsIn)
            done()
          })
        })

        context('with errors', function() {
          it('stops spawning processes after the first failure', function(done) {
            var error = {abc: 123}
            spawnNotifier.on('spawned', function(proc) {
              proc.emit('error', error)
            })
            deploy.deployForEnv(conf, 'staging', [], function(err, args) {
              err.should.be.an.Object
              err.should.eql(error)
              spawnCalls.length.should.eql(1)
              done()
            })
          })
        })
      })
    })
  })
})
