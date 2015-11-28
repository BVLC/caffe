var displayutils = require('../../displayutils')
var XmlDom = require('xmldom')


function XUnitReporter(silent, out, printIntermediateOutput){
  this.out = out || process.stdout
  this.printIntermediateOutput = printIntermediateOutput
  this.silent = silent
  this.stoppedOnError = null
  this.id = 1
  this.total = 0
  this.pass = 0
  this.results = []
  this.startTime = new Date()
  this.endTime = null
}
XUnitReporter.prototype = {
  report: function(prefix, data){
    this.results.push({
      launcher: prefix,
      result: data
    })
    this.display(prefix, data)
    this.total++
    if (data.passed) this.pass++
  },
  finish: function(){
    if (this.silent) return
    this.endTime = new Date()
    this.out.write(this.summaryDisplay())
    this.out.write('\n')
  },
  summaryDisplay: function(){
    var doc = new XmlDom.DOMImplementation().createDocument('','testsuite')

    var rootNode = doc.documentElement
    rootNode.setAttribute('name', 'Testem Tests');
    rootNode.setAttribute('tests', this.total);
    rootNode.setAttribute('failures', this.failures());
    rootNode.setAttribute('timestamp', new Date);
    rootNode.setAttribute('time', this.duration());

    for (var i = 0, len = this.results.length; i < len; i++) {
      var testcaseNode = this.getTestResultNode(doc, this.results[i])
      rootNode.appendChild(testcaseNode)
    }

    return new XmlDom.XMLSerializer().serializeToString(doc.documentElement);
  },
  display: function(prefix, result){
    if (this.silent || !this.printIntermediateOutput || this.out == process.stdout) return
    process.stdout.write(displayutils.resultString(this.id++, prefix, result))
  },
  getTestResultNode: function(document, result){
    var launcher = result.launcher
    result = result.result

    var resultNode = document.createElement('testcase');
    resultNode.setAttribute('classname', launcher);
    resultNode.setAttribute('name', result.name);
    resultNode.setAttribute('time', this._durationFromMs(result.runDuration));

    var error = result.error
    if (error) {
      var failureNode = document.createElement('failure');
      failureNode.setAttribute('name', result.name);
      failureNode.setAttribute('message', error.message);
      if (error.stack)
      {
        var cdata = document.createCDATASection(error.stack)
        failureNode.appendChild(cdata);
      }
      resultNode.appendChild(failureNode)
    }
    return resultNode
  },
  failures: function(){
    return this.total - this.pass
  },
  duration: function(){
    return this._durationFromMs(this.endTime - this.startTime);
  },
  _durationFromMs: function(ms){
    if (ms)
    {
      return (ms / 1000).toFixed(3);
    }
    else
    {
      return 0;
    }
  }
}



module.exports = XUnitReporter
