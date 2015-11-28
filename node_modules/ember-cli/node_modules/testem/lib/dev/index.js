/*

dev_mode_app.js
===============

This is the entry point for development(TDD) mode.

*/

var EventEmitter = require('events').EventEmitter
var Backbone = require('backbone')
var async = require('async')
var Server = require('../server')
var log = require('npmlog')
var AppView = require('./ui/appview')
var HookRunner = require('../hook_runner')
var BrowserRunner = require('../browser_runner')
var Path = require('path')
var fireworm = require('fireworm')
var StyledString = require('styled_string')
var cleanExit = require('../clean_exit')


function App(config, finalizer, cb){
  var self = this

  this.config = config
  this.runners = new Backbone.Collection

  this
    .on('all-test-results', function () {
      var allRunnersComplete = self.runners.all(function (runner) {
        var results = runner.get('results')
        return results && !!results.get('all')
      })
      if (allRunnersComplete) {
        self.emit('all-runners-complete')
      }
    })

  var quiteGracefully = function(err) {
    console.error(err, err.stack)
    self.quit(1, err)
  }

  process.on('uncaughtException', quiteGracefully)

  this.exited = false
  this.paused = false
  this.finalizer = function(code, cb) {
    var fn = finalizer || cleanExit
    process.removeListener('uncaughtException', quiteGracefully)
    fn(code);
    if (cb) {
      cb();
    }
  }

  // a list of connected browser clients
  this.runners.on('remove', function(runner){
    runner.unbind()
  })

  this.configureView()

  this.configure(function(){
    this.server = new Server(config)
    this.server.on('server-start', this.initView.bind(this))
    this.server.on('file-requested', this.onFileRequested.bind(this))
    this.server.on('browser-login', this.onBrowserLogin.bind(this))
    this.server.on('server-error', this.onServerError.bind(this))
    this.server.start(cb)
  })
}

App.prototype = {
  __proto__: EventEmitter.prototype,
  start: function(){},

  _showError: function(titleText, err) {
    var title = StyledString(titleText + '\n ').foreground('red')
    var errMsgs = StyledString('\n' + err.name)
                   .foreground('white')
                   .concat(StyledString('\n' + err.message).foreground('red'))
    this.view.setErrorPopupMessage(title.concat(errMsgs))
    log.log('warn', titleText, {
      name: err.name,
      message: err.message
    })
  },

  initView: function() {
    var self = this
    var view = this.view
    if (this.view.on)
      this.view.on('inputChar', this.onInputChar.bind(this))

    this.on('all-runners-complete', function(){
      self.runPostprocessors()
    })

    self.startOnStartHook(function(err){
      if (err){
        var titleText = 'Error running on_start hook'
        self._showError(titleText, err)
        return
      } else {
        self.startTests(function(){
          self.initLaunchers()
        })
      }
    })
  },
  initLaunchers: function(){
    var config = this.config
    var launch_in_dev = config.get('launch_in_dev')
    var self = this

    config.getLaunchers(function(err, launchers){
      self.launchers = launchers
      launchers.forEach(function(launcher){
        log.info('Launching ' + launcher.name)
        launcher.start()
        if (launcher.runner){
          self.runners.push(launcher.runner)

        }
      })
    })
  },
  configureView: function() {
    var self = this
    this.view = new AppView({
      app: this
    })
    this.view.on('ctrl-c', function(){
      self.quit()
    })
  },
  configure: function(cb){
    var self = this
    var config = self.config
    config.read(function(){
      if (config.get('disable_watching')) {
        if (cb) {
          cb.call(self)
        }

        return
      }

      self.fileWatcher = fireworm('./', {
        ignoreInitial: true,
        skipDirEntryPatterns: []
      })
      var onFileChanged = self.onFileChanged.bind(self)
      self.fileWatcher.on('change', onFileChanged)
      self.fileWatcher.on('add', onFileChanged)
      self.fileWatcher.on('remove', onFileChanged)
      self.fileWatcher.on('emfile', self.onEMFILE.bind(self))

      var fileWatcher = self.fileWatcher
      var watchFiles = config.get('watch_files')
      fileWatcher.clear()
      var confFile = config.get('file')
      if (confFile){
        fileWatcher.add(confFile)
      }
      if (config.isCwdMode()){
        fileWatcher.add('*.js')
      }
      if (watchFiles) {
        fileWatcher.add(watchFiles)
      }
      var srcFiles = config.get('src_files') || '*.js'
      fileWatcher.add(srcFiles)
      var ignoreFiles = config.get('src_files_ignore')
      if (ignoreFiles){
        fileWatcher.ignore(ignoreFiles)
      }
      if (cb) {
        cb.call(self)
      }
    })
  },
  onFileRequested: function(filepath){
    if (this.fileWatcher && !this.config.get('serve_files')){
      this.fileWatcher.add(filepath)
    }
  },
  onFileChanged: function(filepath){
    log.info(filepath + ' changed ('+(this.disableFileWatch ? 'disabled' : 'enabled')+').')
    if (this.disableFileWatch || this.paused) return
    var configFile = this.config.get('file')
    if ((configFile && filepath === Path.resolve(configFile)) ||
      (this.config.isCwdMode() && filepath === process.cwd())){
      // config changed
      this.configure(this.startTests.bind(this))
    }else{
      this.runHook('on_change', {file: filepath}, this.startTests.bind(this))
    }
  },
  onEMFILE: function(){
    var view = this.view
    var text = [
      'The file watcher received a EMFILE system error, which means that ',
      'it has hit the maximum number of files that can be open at a time. ',
      'Luckily, you can increase this limit as a workaround. See the directions below \n \n',
      'Linux: http://stackoverflow.com/a/34645/5304\n',
      'Mac OS: http://serverfault.com/a/15575/47234'
    ].join('')
    view.setErrorPopupMessage(StyledString(text + '\n ').foreground('megenta'))
  },
  onServerError: function(err){
    this.quit(1, err)
  },
  onGeneralWatcherError: function(message){
    log.error('Error from fireworm: ' + message)
  },
  onBrowserLogin: function(browserName, id, client){
    this.connectBrowser(browserName, id, client)
  },
  quit: function(code, err, cb){
    if (this.exited) return

    var self = this
    this.emit('exit')
    this.cleanUpLaunchers(function() {
      self.runExitHook(function() {
        self.cleanupView(function() {
          self.stopServer(function() {
            if (err) console.error(err.stack)
            self.finalizer(code, cb)
            self.exited = true
          })
        })
      })
    })
  },

  stopServer: function(cb) {
    if (!this.server) {
      return cb();
    }

    this.server.stop(cb);
  },

  cleanupView: function(cb) {
    if (!this.view || !this.view.cleanup) {
      return cb();
    }

    this.view.cleanup(cb);
  },

  onInputChar: function(chr, i) {
    var self = this
    if (chr === 'q') {
      log.info('Got keyboard Quit command')
      this.quit()
    }
    else if (i === 13){ // ENTER
      log.info('Got keyboard Start Tests command')
      this.startTests()
    }
    else if (chr === 'p') {
      this.paused = !this.paused
      this.view.renderBottom()
    }
  },
  startTests: function(callback){
    if (this.paused) {
      return
    }
    try{
      var view = this.view
      var runners = this.runners
      var self = this
      this.runPreprocessors(function(err){
        if (err){
          var titleText = 'Error running before_tests hook'
          self._showError(titleText, err);
          return
        }else{
          view.clearErrorPopupMessage()
          runners.forEach(function(runner){
            runner.startTests()
          })
          if (callback) callback()
        }
      })
    }catch(e){
      log.info(e.message)
      log.info(e.stack)
    }
  },
  runPreprocessors: function(callback){
    this.runHook('before_tests', callback)
  },
  runPostprocessors: function(callback){
    this.runHook('after_tests', callback)
  },
  startOnStartHook: function(callback){
    this.onStartProcess = new HookRunner(this.config)
    this.onStartProcess.run('on_start', [], callback)
  },
  runExitHook: function (callback) {
    if(this.onStartProcess) {
      this.onStartProcess.stop()
    }
    this.runHook('on_exit', callback)
  },
  runHook: function(/*hook, data..., callback*/){
    var hook = arguments[0]
    var callback = arguments[arguments.length-1]
    var data = arguments.length > 2 ? arguments[1] : {}
    var runner = new HookRunner(this.config)
    var self = this
    log.info("Hook "+hook+" started")
    this.disableFileWatch = true
    runner.run(hook, data, function(err){
      log.info("Hook "+hook+" finished")
      self.disableFileWatch = false
      if (callback) { callback(err) }
    })
  },
  removeBrowser: function(browser){
    this.runners.remove(browser)
  },
  connectBrowser: function(browserName, id, client){
    var existing = this.runners.find(function(runner){
      return runner.pending && runner.get('name') === browserName
    })
    if (existing){
      clearTimeout(existing.pending)
      existing.set('socket', client)
      return existing
    }else{
      var browser = new BrowserRunner({
        name: browserName,
        socket: client
      })
      var self = this
      browser.on('disconnect', function(){
        browser.pending = setTimeout(function(){
          self.removeBrowser(browser)
        }, 1000)
      })
      browser.on('all-test-results', function(results, browser){
        self.emit('all-test-results', results, browser)
      })
      browser.on('top-level-error', function(msg, url, line){
        self.emit('top-level-error', msg, url, line)
      })
      this.runners.push(browser)
      return browser
    }
  },
  cleanUpLaunchers: function(callback){
    if (!this.launchers){
      if (callback) callback()
      return
    }
    async.forEach(this.launchers, function(launcher, done){
      if (launcher && launcher.process){
        launcher.kill('SIGTERM', done)
      }else{
        done()
      }
    }, callback)
  }
}

module.exports = App
