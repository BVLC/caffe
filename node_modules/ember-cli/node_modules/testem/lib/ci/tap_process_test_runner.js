var TapConsumer = require('../tap_consumer')

function TapProcessTestRunner(launcher, reporter){
  this.launcher = launcher
  this.tapConsumer = new TapConsumer()
  this.reporter = reporter
}
TapProcessTestRunner.prototype = {
  start: function(onFinish){
    this.onFinish = onFinish
    this.launcher.start()
    this.launcher.process.stdout.pipe(this.tapConsumer.stream)
    this.launcher.once('processError', this.onProcessError.bind(this))

    this.tapConsumer.on('test-result', this.onTestResult.bind(this))
    this.tapConsumer.on('all-test-results', this.onAllTestResults.bind(this))
  },
  onTestResult: function(test){
    this.reporter.report(this.launcher.name, test)
  },
  onAllTestResults: function(err, count){
    this.wrapUp()
  },
  wrapUp: function(){
    this.launcher.kill(null, function(){
      this.onFinish()
    }.bind(this))
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

module.exports = TapProcessTestRunner
