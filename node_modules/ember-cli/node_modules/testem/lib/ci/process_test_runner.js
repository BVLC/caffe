function ProcessTestRunner(launcher, reporter){
  this.launcher = launcher
  this.reporter = reporter
}
ProcessTestRunner.prototype = {
  start: function(onFinish){
    this.onFinish = onFinish
    this.launcher.start()
    this.launcher.once('processExit', this.onProcessExit.bind(this))
    this.launcher.once('processError', this.onProcessError.bind(this))
  },
  onProcessExit: function(code, stdout, stderr){
    var result = {
      passed: code === 0,
      name: this.launcher.commandLine()
    }
    if (!result.passed){
      result.error = {
        stdout: stdout,
        stderr: stderr
      }
    }
    this.reporter.report(this.launcher.name, result)
    this.onFinish()
  },
  onProcessError: function(err, stdout, stderr){
    var result = {
      passed: false,
      name: this.launcher.commandLine(),
      error: {
        err: err,
        stdout: stdout,
        stderr: stderr
      }
    }
    this.reporter.report(this.launcher.name, result)
    this.onFinish()
  }
}

module.exports = ProcessTestRunner
