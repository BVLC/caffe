var assert = require('assert')
var fs = require('fs')
var async = require('async')
var path = require('path')
var File = require('./file')
var debounce = require('lodash').debounce
var minimatch = require('minimatch')
var is = require('is-type')
var hasOwnProperty = Object.prototype.hasOwnProperty

function Dir(dirpath, sink, wantDir){
  assert(is.string(dirpath))
  assert(sink)
  assert(is.function(sink.emit))
  assert(wantDir == null || is.function(wantDir))

  this.path = dirpath
  this.sink = sink
  this.wantDir = wantDir || function(){ return true }
  this.entries = {}
  this.watcher = null
  this.sink.emit('add', this.path)
  this.update = debounce(this._update.bind(this, false), 200)
  this.forceUpdate = debounce(this._update.bind(this, true), 200)
}

Dir.prototype = {
  _isEntry: function(filename){
    return hasOwnProperty.call(this.entries, filename)
  },
  _update: function(force, doneUpdate){
    assert(is.boolean(force))
    assert(doneUpdate == null || is.function(doneUpdate))
    doneUpdate = doneUpdate || function(){}
    var self = this
    
    if (!this.watcher){
      this._watch()
    }

    fs.readdir(this.path, function(err, entryNames){
      if (err){
        if (err.code === 'ENOENT'){
          // ignore, this means the directory has been
          // removed, but the parent node should
          // handle the destroy
        }else{
          // unexpected error, emit as event
          self.sink.emit('error', err)
        }
        return
      }

      async.eachLimit(entryNames, 400, function(entryName, next){
        var entry = self.entries[entryName]
        
        if (entry){
          if (entry.isDirectory()){
            // do nothing for existing directories
            if (force){
              entry.forceUpdate(next)
            }else{
              next()
            }
          }else{ // is file
            entry.update(next)
          }
        }else{ // is new
          self._addNewEntry(entryName, next)
        }

      }, function(){

        // detect removed entries
        for (var entryName in self.entries){
          if (entryNames.indexOf(entryName) === -1){
            // entry was removed
            var entry = self.entries[entryName]
            entry.destroy()
            delete self.entries[entryName]
          }
        }

        doneUpdate()

      })
    })
  },
  _addNewEntry: function(entryName, callback){
    var self = this
    var entryPath = path.join(self.path, entryName)
    if (self.wantDir && !self.wantDir(entryPath)){
      return callback()
    }
    fs.stat(entryPath, function(err, stat){
      if (err){
        if (err.code === 'ENOENT'){
          // ignore - it was a fleeting file?
        }else{
          self.sink.emit('error', err)
        }
        return
      }
      
      if (stat.isDirectory()){
        var dir = self.entries[entryName] = new Dir(
          entryPath, self.sink, self.wantDir)
        dir.update(callback)
      }else{
        if (!self._isEntry(entryName)){
          self.entries[entryName] = new File(
            entryPath, self.sink, stat)
        }
        callback(null)
      }
    })
  },
  _watch: function(){
    var self = this
    try{
      this.watcher = fs.watch(this.path, function(evt, filename){
        if (evt === 'change' && self._isEntry(filename)){
          self.entries[filename].update()
        }else{
          self.update()
        }
      })
      this.watcher.on('error', function(err){
        if (err.code === 'EPERM') return
        self.sink.emit('error', err)
      })
    }catch(e){
      if (e.code === 'ENOENT'){
        this.destroy()
        return
      }else{
        throw new Error(e.message + ' - ' + self.code + ' - ' + self.path)
      }
    }
  },
  isDirectory: function(){
    return true
  },
  destroy: function(){
    if (this.watcher){
      this.watcher.close()
    }
    for (var entryName in this.entries){
      var entry = this.entries[entryName]
      entry.destroy()
    }
    this.sink.emit('remove', this.path)
  },
  numWatchers: function(){
    var sum = this.watcher ? 1 : 0
    for (var entryName in this.entries){
      var entry = this.entries[entryName]
      sum += entry.numWatchers()
    }
    return sum
  }
}

module.exports = Dir
