var childProcess = require('child_process')
var crossSpawn = require('cross-spawn-async')
var EventEmitter = require('events').EventEmitter
var log = require('npmlog')
var fileutils = require('./fileutils')
var async = require('async')
var ProcessRunner = require('./process_runner')
var template = require('./strutils').template

function Launcher(name, settings, config){
  this.name = name
  this.config = config
  this.settings = settings
  this.setupDefaultSettings()
  this.id = settings.id || String(Math.floor(Math.random() * 10000))
}

Launcher.prototype = {
  __proto__: EventEmitter.prototype,
  killTimeout: 5000,
  setupDefaultSettings: function(){
    var settings = this.settings
    if (settings.protocol === 'tap' && !('hide_stdout' in settings)){
      settings.hide_stdout = true
    }
  },
  isProcess: function(){
    return this.settings.protocol !== 'browser'
  },
  protocol: function(){
    return this.settings.protocol || 'process'
  },
  commandLine: function(){
    if (this.settings.command){
      return '"' + this.settings.command + '"'
    }else if (this.settings.exe){
      return '"' + this.settings.exe +
        ' ' + this.getArgs().join(' ') + '"'
    }
  },
  start: function(){
    if (this.isProcess()){
      var self = this
      self.runner = new ProcessRunner({
        launcher: self
      })
    }else{
      this.launch()
    }
  },
  getUrl: function(){
    return this.config.get('url') + this.id
  },
  launch: function(cb){
    var self = this
    var settings = this.settings
    this.kill('SIGTERM', function(){
      if (settings.setup){
        settings.setup.call(self, self.config, function(){
          self.doLaunch(cb)
        })
      }else{
        self.doLaunch(cb)
      }
    })

  },
  doLaunch: function(cb){
    var id = this.id
    var settings = this.settings
    var self = this
    var options = {}
    if (settings.cwd) {
      options.cwd = settings.cwd
    }
    if (settings.exe){

      function spawn(exe, useCrossSpawn){
        args = self.template(args, id)
        log.info('spawning: ' + exe + ' - ' + JSON.stringify(args))
        if (useCrossSpawn) {
          self.process = crossSpawn(exe, args, options)
        }
        else {
          self.process = childProcess.spawn(exe, args, options)
        }
        self.process.once('close', self.onClose.bind(self))
        self.process.once('error', self.onError.bind(self))
        self.stdout = ''
        self.stderr = ''
        self.process.stdout.on('data', function(chunk){
          self.stdout += chunk
        })
        self.process.stderr.on('data', function(chunk){
          self.stderr += chunk
        })
        self.emit('processStarted', self.process)
        if (cb) {
          cb(self.process)
        }
      }

      var args = self.getArgs()

      if (Array.isArray(settings.exe)){
        async.detectSeries(settings.exe, self.exeExists, function(found) {
          // since we found executable file we don't need array anymore in this run
          self.settings.exe = found
          spawn(found, settings.useCrossSpawn)
        })
      }else{
        spawn(settings.exe, settings.useCrossSpawn)
      }

    }else if (settings.command){
      var cmd = this.template(settings.command, id)
      log.info('cmd: ' + cmd)
      this.process = childProcess.exec(cmd, options, function(err, stdout, stderr){
        self.stdout = stdout
        self.stderr = stderr
      })
      this.process.on('close', self.onClose.bind(self))
      this.process.on('error', self.onError.bind(self))
      self.emit('processStarted', self.process)
      if (cb) {
        cb(self.process)
      }
    }
  },
  getArgs: function(){
    var settings = this.settings
    var url = this.config.get('url') + this.id
    var args = [url]
    if (settings.args instanceof Array)
      args = settings.args.concat(args)
    else if (settings.args instanceof Function){
      args = settings.args.call(this, this.config)
    }
    return args
  },
  template: function(thing, id){
    if (Array.isArray(thing)){
      return thing.map(this.template, this)
    }else{
      var params = {
        url: this.config.get('url') + id,
        port: this.config.get('port')
      }
      return template(thing, params)
    }
  },
  exeExists: function(exe, cb){
    fileutils.fileExists(exe, function(yes){
      if (yes) return cb(true)
      else fileutils.which(exe, function(yes){
        if (yes) return cb(true)
        else fileutils.where(exe, cb)
      })
    })
  },
  onClose: function(code){
    if (!this.process) {
      return;
    }
    log.warn(this.name + ' closed', code)
    this.process = null
    this.exitCode = code
    this.emit('processExit', code, this.stdout, this.stderr)
  },
  onError: function(error){
    log.warn(this.name + ' errored', error)
    this.process = null
    this.exitCode = 1
    this.emit('processError', 1, error, this.stdout, this.stderr)
  },
  kill: function(sig, cb){
    if (!this.process){
      log.info('Process ' + this.name + ' already killed.')
      if(cb) {
        cb(this.exitCode)
      }
      return
    }
    var launcherProcess = this.process
    var self = this
    sig = sig || 'SIGTERM'

    var exited = false
    launcherProcess.on('close', function(code, sig) {
      if (exited) {
        return;
      }
      exited = true
      if (self._killTimer) {
        clearTimeout(self._killTimer);
        self._killTimer = null;
      }
      log.info('Process ' + self.name + ' terminated.', code, sig)
      launcherProcess.stdout.removeAllListeners()
      launcherProcess.stderr.removeAllListeners()
      launcherProcess.removeAllListeners()
      if (cb) {
        cb(code)
      }
    })
    launcherProcess.on('error', function(err) {
      log.error('Error killing process ' + self.name + '.', err)
    })
    this._killTimer = setTimeout(this.onKillTimeout.bind(this), this.killTimeout);
    if (this.settings.useCrossSpawn) {
      childProcess.exec('taskkill /pid ' + this.process.pid + ' /T');
    }
    else {
      this.process.kill(sig)
    }
  },

  onKillTimeout: function() {
    log.warn('Process ' + this.name + ' not terminated in ' + this.killTimeout + 'ms.')
    if (this.settings.useCrossSpawn) {
      childProcess.exec('taskkill /pid ' + this.process.pid + ' /T /F');
    }
    else {
      this.process.kill('SIGKILL')
    }
  }
}

module.exports = Launcher
