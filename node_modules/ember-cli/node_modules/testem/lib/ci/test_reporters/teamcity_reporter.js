var strutils = require('../../strutils')

function TeamcityReporter(silent, out){
  this.out = out || process.stdout
  this.silent = silent
  this.stoppedOnError = null
  this.id = 1
  this.total = 0
  this.pass = 0
  this.startTime = new Date()
  this.endTime = null
}
TeamcityReporter.prototype = {
  report: function(prefix, data){
    this.out.write("##teamcity[testStarted name='" + this._namify(prefix, data) + "']\n")
    this._display(prefix, data)
    this.total++
    if (data.passed) this.pass++
  },
  finish: function(){
    if (this.silent) return
    this.endTime = new Date()
    this.out.write('\n\n')
    this.out.write("##teamcity[testSuiteFinished name='mocha.suite' duration='" + this._duration() + "']\n")
    this.out.write('\n\n')
  },
  _display: function(prefix, result){
    if (this.silent) return
    if (!result.passed) {
      this.out.write("##teamcity[testFailed name='" + this._namify(prefix, result) + "' message='"+ escape(result.error.message) +"' details='" + escape(result.error.stack) + "']\n")
    }
    this.out.write("##teamcity[testFinished name='" + this._namify(prefix, result) + "']\n")

  },
  _namify: function(prefix, result) {
    var line = (prefix ? (prefix + ' - ') : '') +
      result.name.trim()
    return escape(line)
  },
  _duration: function(){
    return Math.round((this.endTime - this.startTime))
  }
}


/**
 * Borrowed from https://github.com/travisjeffery/mocha-teamcity-reporter
 * Escape the given `str`.
 */

function escape(str) {
  if (!str) return ''
  return str
    .toString()
    .replace(/\|/g, "||")
    .replace(/\n/g, "|n")
    .replace(/\r/g, "|r")
    .replace(/\[/g, "|[")
    .replace(/\]/g, "|]")
    .replace(/\u0085/g, "|x")
    .replace(/\u2028/g, "|l")
    .replace(/\u2029/g, "|p")
    .replace(/'/g, "|'")
}

module.exports = TeamcityReporter
