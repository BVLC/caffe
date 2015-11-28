var fs = require('fs')
var path = require('path')
var mkdirp = require('mkdirp')
var PassThrough = require('stream').PassThrough

function ReportFile(reportFile, out) {
  this.file = reportFile

  this.outputStream = new PassThrough()

  mkdirp.sync(path.dirname(path.resolve(reportFile)))

  var fileStream = fs.createWriteStream(reportFile, { flags: 'w+' })

  this.outputStream.on('data', function(data) {
    out.write(data)
    fileStream.write(data)
  })

  this.outputStream.on('end', function(data) {
    out.end(data)
    fileStream.end(data)
  })

  this.fileStream = fileStream
  this.stream = this.outputStream
}

module.exports = ReportFile
