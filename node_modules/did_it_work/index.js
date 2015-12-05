var child_process = require('child_process')

module.exports = function(arg1, arg2){
  return new Process(arg1, arg2)
}

function Process(arg1, arg2){
  if (arg2){
    this.exe = arg1
    this.args = arg2
  }else{
    this.command = arg1
  }
  this.opts = {}
  this.stdout = ''
  this.stderr = ''
  this.boundListeners = {}
  this.successDetermined = false
  process.nextTick(function(){
    this.start()
  }.bind(this))
}

Process.prototype = {

  /* public/documented methods */
  good: function(callback){
    this.opts.good = callback
    return this
  },

  bad: function(callback){
    this.opts.bad = callback
    return this
  },

  goodIfMatches: function(pattern, timeout){
    this.opts.goodIfMatches = pattern
    this.opts.goodIfMatchesTimeout = timeout
    return this
  },

  badIfMatches: function(pattern, timeout){
    this.opts.badIfMatches = pattern
    this.opts.badIfMatchesTimeout = timeout
    return this
  },

  complete: function(callback){
    this.opts.complete = callback
    return this
  },

  options: function(options){
    this.opts.options = options
    return this
  },

  /* non-documented method below */

  start: function(){
    this.opts.__proto__ = {
      // defaults so I don't have to guard
      good: function(){},
      bad: function(){},
      complete: function(){}
    }
    
    if (this.opts.goodIfMatchesTimeout){
      setTimeout(this.goodIfMatchesTimedOut.bind(this), 
        this.opts.goodIfMatchesTimeout)
    }
    if (this.opts.badIfMatchesTimeout){
      setTimeout(this.badIfMatchesTimedOut.bind(this), 
        this.opts.badIfMatchesTimeout)
    }

    this.process = this.createProcess()

    this.boundListeners.onStdoutData = this.onStdoutData.bind(this)
    this.process.stdout.on('data', this.boundListeners.onStdoutData)

    this.boundListeners.onStderrData = this.onStderrData.bind(this)
    this.process.stderr.on('data', this.boundListeners.onStderrData)

    this.process.once('close', this.onProcessClose.bind(this))
  },

  createProcess: function(){
    if (this.exe){
      return child_process.spawn(this.exe, this.args || [], this.opts.options)
    }else{
      return child_process.exec(this.command, this.opts.options)
    }
  },

  onStdoutData: function(data){
    data = String(data)
    this.stdout += data
    if (this.foundGoodMatch(data)){
      this.determine('good', data)
    }
    if (this.foundBadMatch(data)){
      this.determine('bad', new Error('Found bad match(' + this.opts.badIfMatches + '): ' + data))
    }
  },

  foundGoodMatch: function(stdout){
    var pattern = this.opts.goodIfMatches
    if (!pattern) return false
    var lines = stdout.split('\n')
    return lines.some(function(line){
      return this.patternMatches(line, pattern)
    }, this)
  },

  foundBadMatch: function(stdout){
    var pattern = this.opts.badIfMatches
    if (!pattern) return false
    var lines = stdout.split('\n')
    return lines.some(function(line){
      return !!line.match(pattern)
    }, this)
  },

  patternMatches: function(line, pattern){
    if (typeof pattern === 'string'){
      return line.indexOf(pattern) !== -1
    }else{ // regex
      return !!line.match(pattern)
    }
  },

  onStderrData: function(data){
    this.stderr += String(data)
  },

  onProcessClose: function(code){
    if (code !== 0){
      this.determine('bad', new Error(this.stderr))
    }
    this.opts.complete(this.error, this.stdout, this.stderr)
  },

  goodIfMatchesTimedOut: function(){
    var err = new Error('Timed out without seeing ' + 
      this.opts.goodIfMatches)
    this.determine('bad', err)
  },

  badIfMatchesTimedOut: function(){
    this.determine('good', this.stdout)
  },

  determine: function(type){
    if (this.successDetermined) return
    var args = Array.prototype.slice.call(arguments, 1)
    // save error for complete call
    if (type === 'bad'){
      this.error = args[0]
    }
    args.push(this.stdout)
    args.push(this.stderr)
    this.opts[type].apply(null, args)
    this.successDetermined = true
    this.cleanupListeners()
  },

  cleanupListeners: function(){
    this.process.stdout.removeListener('data', 
      this.boundListeners.onStdoutData)
    this.process.stderr.removeListener('data',
      this.boundListeners.onStderrData)
  },

  _kill: function(){
    var args = Array.prototype.slice.apply(arguments)
    var sig = 'SIGTERM'
    if (typeof args[0] === 'string'){
      sig = args.shift()
    }
    var callback = args.shift()
    if (callback) this.process.once('exit', callback)
    this.process.kill(sig)
  },

  kill: function(){
    var args = arguments
    
    var doit = function(){
      this._kill.apply(this, args)
    }.bind(this)

    if (this.process){
      doit()
    }else{
      process.nextTick(doit)
    }
  }

}
