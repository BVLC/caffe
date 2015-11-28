// Method to format test results.
var strutils = require('./strutils')

function resultDisplay(id, prefix, result) {
  var line = (prefix ? (prefix + ' - ') : '') +
    result.name.trim()
  return (result.passed ? 'ok ' : 'not ok ') + id + ' ' + line
}

function yamlDisplay(err, logs) {
  var failed = Object.keys(err || {})
    .filter(function(key){
      return key !== 'passed'
    })
    .map(function(key){
      return key + ': >\n' + strutils.indent(String(err[key]))
    })
  if(logs){
      var testLogs = ["Log: |"].concat(logs.map(function(log){return strutils.indent(String(log))}))
  } else {
      var testLogs = []
  }
  return strutils.indent([
    '---',
    strutils.indent(failed.concat(testLogs).join('\n')),
    '...'].join('\n'))
}

function resultString(id, prefix, result) {
  var string = resultDisplay(id, prefix, result) + '\n'
  if (result.error || result.logs && result.logs.length){
    string += yamlDisplay(result.error, result.logs) + '\n'
  }
  return string
}

exports.resultString = resultString
