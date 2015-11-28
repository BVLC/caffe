var indent = require('../../strutils').indent
var printf = require('printf')

function DotReporter(silent, out){
  this.out = out || process.stdout
  this.silent = silent
  this.stoppedOnError = null
  this.id = 1
  this.total = 0
  this.pass = 0
  this.results = []
  this.startTime = new Date()
  this.endTime = null
  this.out.write('\n')
  this.out.write('  ')
}
DotReporter.prototype = {
  report: function(prefix, data){
    this.results.push({
      launcher: prefix,
      result: data
    })
    this.display(prefix, data)
    this.total++
    if (data.passed) this.pass++
  },

  display: function(prefix, result){
    if (this.silent) return
    if (result.passed) {
      this.out.write('.')
    } else {
      this.out.write('F')
    }
  },
  finish: function(){
    if (this.silent) return
    this.endTime = new Date()
    this.out.write('\n\n')
    this.out.write(this.summaryDisplay())
    this.out.write('\n\n')
    this.displayErrors()
  },
  displayErrors: function(){
    this.results.forEach(function(data, idx){
      var result = data.result
      var error = result.error
      if (!error) return

      printf(this.out, '%*d) [%s] %s\n', idx+1, 3, data.launcher, result.name)

      if (error.message) printf(this.out, '     %s\n', error.message)

      if ('expected' in error && 'actual' in error) {
        printf(this.out, '\n' +
               '     expected: %O\n' +
               '       actual: %O\n', error.expected, error.actual)
      }

      if (error.stack) printf(this.out, '\n%s', indent(error.stack, 5))

      this.out.write('\n')
    }, this)
  },
  summaryDisplay: function(){
    return printf('  %d tests complete (%d ms)', this.total, this.duration())
  },
  duration: function(){
    return Math.round((this.endTime - this.startTime))
  }
}



module.exports = DotReporter
