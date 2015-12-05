var assert = require('assert')
var fs = require('fs')
var debounce = require('lodash').debounce
var is = require('is-type')

function File(filepath, sink, stat){
  assert(is.string(filepath))
  assert(stat)
  assert(stat.isFile())
  assert(sink)
  assert(is.function(sink.emit))

  this.path = filepath
  this.stat = stat
  this.sink = sink
  this.sink.emit('add', this.path)
  this.update = debounce(this._update.bind(this), 200, {
    leading: true,
    trailing: false
  })
  this.forceUpdate = this.update
}

File.prototype = {
  _update: function(callback){
    assert(callback == null || is.function(callback))
    
    callback = callback || function(){}
    var self = this
    var prevStat = this.stat
    fs.stat(this.path, function(err, stat){
      self.stat = stat
      if (err){
        if (err.code === 'ENOENT'){
          // file no longer exists
          // but ignore because parent node will
          // take care of clean up
          return callback()
        }else{
          // unexpected error, emit as an event
          self.sink.emit('error', err)
          return callback(err)
        }
      }
      assert(prevStat != null, 'File should always be initialied with stat')
      if (stat.mtime.getTime() > prevStat.mtime.getTime()){
        self.sink.emit('change', self.path)
      }else{
        // remained the same
      }
      callback(null)
    })
  },
  isDirectory: function(){
    return false
  },
  destroy: function(){
    this.sink.emit('remove', this.path)
  },
  numWatchers: function(){
    return 0
  }
}

module.exports = File