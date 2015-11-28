var EventEmitter = require('events').EventEmitter
var path = require('path')
var fs = require('fs')
var util = require('util')

function Walker (dir, filter) {
  EventEmitter.call(this)
  this.path = path.resolve(dir)
  this.filter = filter
  this.pending = 0
}
util.inherits(Walker, EventEmitter)

Walker.prototype.start = function () {
  this.visit(this.path)
  return this
}

Walker.prototype.visit = function (item) {
  this.pending++
  var self = this

  fs.lstat(item, function (err, stat) {
    if (err) {
      self.emit('error', err, item, stat)
      return self.finishItem()
    }

    if (self.filter && !self.filter(item, stat)) return self.finishItem()

    if (!stat.isDirectory()) {
      self.emit('data', item, stat)
      return self.finishItem()
    }

    fs.readdir(item, function (err, items) {
      if (err) {
        self.emit('error', err, item, stat)
        return self.finishItem()
      }

      self.emit('data', item, stat)
      items.forEach(function (part) {
        self.visit(path.join(item, part))
      })
      self.finishItem()
    })
  })
  return this
}

Walker.prototype.finishItem = function () {
  this.pending -= 1
  if (this.pending === 0) this.emit('end')
  return this
}

module.exports = Walker
