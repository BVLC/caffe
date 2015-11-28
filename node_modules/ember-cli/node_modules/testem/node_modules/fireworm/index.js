var EventEmitter = require('events').EventEmitter
var minimatch = require('minimatch')
var flatten = require('lodash').flatten
var Dir = require('./lib/dir')
var matchesBeginning = require('./lib/matches_beginning')
var path = require('path')

function Fireworm(dirpath, options){
  if (!(this instanceof Fireworm)){
    return new Fireworm(dirpath, options)
  }

  this.options = options = options || {}
  
  this.patterns = {}
  this.ignores = {}

  if (options.ignoreInitial){
    this.suppressEvents = true
  }
  
  options.skipDirEntryPatterns = options.skipDirEntryPatterns || 
    ['node_modules', 'bower_components', '.*']
  
  var sink = new EventEmitter

  this.dir = new this.Dir(dirpath, sink, this.wantDir.bind(this))

  sink
    .on('add', this._onAdd.bind(this))
    .on('change', this._onChange.bind(this))
    .on('remove', this._onRemove.bind(this))
    .on('error', this._onError.bind(this))

  this.dir.update(function(){
    this.suppressEvents = false
  }.bind(this))
  
}

Fireworm.prototype = {
  __proto__: EventEmitter.prototype,
  Dir: Dir, // to allow injection in tests
  wantDir: function(dirpath){
    var entryName = path.basename(dirpath)
    var skip = this.options.skipDirEntryPatterns.some(function(pattern){
      return minimatch(entryName, pattern)
    })
    if (skip) return false
    return Object.keys(this.patterns).some(function(pattern){
      return matchesBeginning(dirpath, pattern)
    })
  },
  add: function(){
    var args = flatten(arguments)
    var hadNew = false
    for (var i = 0; i < args.length; i++){
      var pattern = path.normalize(args[i])
      if (!this.patterns[pattern]){
        this.patterns[pattern] = true
        hadNew = true
      }
    }
    if (hadNew){
      this.dir.forceUpdate()
    }
  },
  ignore: function(){
    var args = flatten(arguments)
    for (var i = 0; i < args.length; i++){
      var pattern = path.normalize(args[i])
      if (!this.ignores[pattern]){
        this.ignores[pattern] = true
      }
    }
  },
  clear: function(){
    this.patterns = {}
    this.ignores = {}
  },
  _onAdd: function(filepath){
    if (this.suppressEvents) return
    if (this._matches(filepath)){
      this.emit('add', filepath)
    }
  },
  _onRemove: function(filepath){
    if (this.suppressEvents) return
    if (this._matches(filepath)){
      this.emit('remove', filepath)
    }
  },
  _onChange: function(filepath){
    if (this.suppressEvents) return
    if (this._matches(filepath)){
      this.emit('change', filepath)
    }
  },
  _onError: function(err){
    this.emit('error', err)
  },
  _matches: function(filepath){
    return Object.keys(this.patterns).some(function(pattern){
      return minimatch(filepath, pattern)
    }) && !Object.keys(this.ignores).some(function(pattern){
      return minimatch(filepath, pattern)
    })
  },
  numWatchers: function(){
    return this.dir.numWatchers()
  }
}

module.exports = Fireworm
