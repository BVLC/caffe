var Server = require('../server')
var EventEmitter = require('events').EventEmitter
var async = require('async')
var BrowserTestRunner = require('./browser_test_runner')
var ProcessTestRunner = require('./process_test_runner')
var TapProcessTestRunner = require('./tap_process_test_runner')
var test_reporters = require('./test_reporters')
var Process = require('did_it_work')
var HookRunner = require('../hook_runner')
var log = require('npmlog')
var cleanExit = require('../clean_exit')
var isa = require('../isa')
var ReportFile = require('./report_file')

function App(config, finalizer){
  this.exited = false
  this.config = config
  this.stdoutStream = config.get('stdout_stream') || process.stdout
  this.server = new Server(this.config)
  this.cleanExit = finalizer || cleanExit
  this.Process = Process
  this.hookRunners = {}
  this.results = []
  this.reportFileName = this.config.get('report_file')
  this.reportFileStream = this.initReportFileStream(this.reportFileName)
  this.reporter = this.initReporter(this.config.get('reporter'), this.reportFileStream)
  if (!this.reporter){
    console.error('Test reporter `' + this.config.get('reporter') + '` not found.')
    this.cleanExit(1)
  }
}

App.prototype = {
  __proto__: EventEmitter.prototype,
  initReportFileStream: function(path) {
    if(path) {
      this.reportFile = new ReportFile(path, this.stdoutStream)
      return this.reportFile.stream
    } else {
      return this.stdoutStream
    }

  },
  initReporter: function(reporter, stream){
    if (isa(reporter, String)){
      var TestReporter = test_reporters[reporter]
      if (!TestReporter){
        return null
      }
      if (reporter == 'xunit') {
        return new TestReporter(false, stream, this.config.get('xunit_intermediate_output'))
      }
      else {
        return new TestReporter(false, stream)
      }
    } else {
      return reporter
    }
  },
  start: function(){
    log.info('Starting ci')
    async.series([
      this.addSignalListeners.bind(this),
      this.startServer.bind(this),
      this.runHook.bind(this, 'on_start'),
      this.runHook.bind(this, 'before_tests'),
      this.createRunners.bind(this),
      this.registerSocketConnect.bind(this),
      this.startClock.bind(this),
      this.runTheTests.bind(this),
      this.runHook.bind(this, 'after_tests'),
      this.runHook.bind(this, 'on_exit')
    ], this.wrapUp.bind(this))
  },
  startServer: function(callback){
    log.info('Starting server')
    this.server.start(callback)
  },
  runHook: function(/*hook, [data], callback*/){
    var hook = arguments[0]
    var callback = arguments[arguments.length-1]
    var data = arguments.length > 2 ? arguments[1] : {}
    var runner = this.hookRunners[hook] = new HookRunner(this.config, this.Process)
    runner.run(hook, data, callback)
  },
  registerSocketConnect: function(callback){
    this.server.on('browser-login', this.onBrowserLogin.bind(this))
    callback(null)
  },
  onBrowserLogin: function(browser, id, socket){
    this.runners.forEach(function(runner){
      if (runner.tryAttach){
        runner.tryAttach(browser, id, socket)
      }
    })
  },
  createRunners: function(callback){
    var reporter = this.reporter
    var self = this
    this.config.getLaunchers(function(err, launchers){
      if (err) {
        return callback(err)
      }
      self.runners = launchers.map(function(launcher){
        return self.createTestRunner(launcher, reporter)
      })
      callback(null)
    })
  },
  getRunnerFactory: function(launcher){
    var protocol = launcher.protocol()
    switch(protocol){
      case 'process':
        return ProcessTestRunner
      case 'browser':
        return BrowserTestRunner
      case 'tap':
        return TapProcessTestRunner
      default:
        throw new Error("Don't know about " + protocol + " protocol.")
    }
  },
  createTestRunner: function(launcher, reporter){
    return new (this.getRunnerFactory(launcher))(launcher, reporter)
  },
  startClock: function(callback){
    var self = this
    var timeout = this.config.get('timeout')
    if (timeout){
      this.timeoutID = setTimeout(function(){
        self.wrapUp(new Error('Timed out after ' + timeout + 's'))
      }, timeout * 1000)
    }
    callback(null)
  },
  runTheTests: function(callback){
    var self = this
    var limit = this.config.get('parallel')
    async.eachLimit(this.runners, limit, function(runner, next){
      runner.start(next)
    }, callback)
  },
  wrapUp: function(err){
    if (this.timeoutID) {
      clearTimeout(this.timeoutID)
      this.timeoutID = null
    }
    if (err){
      this.reporter.report(null, {
        passed: false,
        name: err.name || 'unknown error',
        error: {
          message: err.message
        }
      })
    }
    this.reporter.finish()
    this.emit('tests-finish')
    this.stopHookRunners()
    async.series([
      this.cleanUpLaunchers.bind(this),
      this.stopServer.bind(this),
      this.removeSignalListeners.bind(this)
    ], this.exit.bind(this))
  },

  stopServer: function(callback){
    this.server.stop(callback)
  },

  stopHookRunners: function(){
    for (var runner in this.hookRunners){
      this.hookRunners[runner].stop()
    }
  },

  getExitCode: function(){
    if (this.reporter.total > this.reporter.pass)
      return 1
    if (this.reporter.total === 0 && this.config.get('fail_on_zero_tests'))
      return 1
    return 0
  },

  exit: function(){
    if (this.exited) return
    this.cleanExit(this.getExitCode())
    this.exited = true
  },

  addSignalListeners: function(callback) {
    this._boundSigInterrupt = function() {
      this.wrapUp(new Error('Received SIGINT signal'))
    }.bind(this)
    process.on('SIGINT', this._boundSigInterrupt)

    this._boundSigTerminate = function() {
      this.wrapUp(new Error('Received SIGTERM signal'))
    }.bind(this)
    process.on('SIGTERM', this._boundSigTerminate)

    callback()
  },

  removeSignalListeners: function(callback) {
    if (this._boundSigInterrupt) {
      process.removeListener('SIGINT', this._boundSigInterrupt)
    }
    if (this._boundSigTerminate) {
      process.removeListener('SIGTERM', this._boundSigTerminate)
    }
    callback()
  },

  cleanUpLaunchers: function(callback){
    if (!this.runners){
      if (callback) callback()
      return
    }
    var launchers = this.runners.map(function(runner) {
      return runner.launcher
    })
    async.forEach(launchers, function(launcher, done){
      if (launcher && launcher.process){
        launcher.kill('SIGTERM', done)
      }else{
        done()
      }
    }, callback)
  }
}

module.exports = App
