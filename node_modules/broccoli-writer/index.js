var quickTemp = require('quick-temp')
var Promise = require('rsvp').Promise


module.exports = Writer
function Writer () {}

Writer.prototype.read = function (readTree) {
  var self = this
  quickTemp.makeOrRemake(this, 'tmpDestDir')
  return Promise.resolve(this.write(readTree, this.tmpDestDir))
    .then(function () {
      return self.tmpDestDir
    })
}

Writer.prototype.cleanup = function () {
  quickTemp.remove(this, 'tmpDestDir')
}
