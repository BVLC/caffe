// Copyright 2014, 2015 Simon Lydell
// X11 (“MIT”) Licensed. (See LICENSE.)

var leftPad = require("left-pad")

function get(options, key, defaultValue) {
  return (key in options ? options[key] : defaultValue)
}

function lineNumbers(code, options) {
  var getOption = get.bind(null, options || {})
  var transform = getOption("transform", Function.prototype)
  var padding   = getOption("padding", " ")
  var before    = getOption("before", " ")
  var after     = getOption("after", " | ")
  var start     = getOption("start", 1)
  var isArray   = Array.isArray(code)
  var lines     = (isArray ? code : code.split("\n"))
  var end       = start + lines.length - 1
  var width     = String(end).length
  var numbered  = lines.map(function(line, index) {
    var number  = start + index
    var params  = {before: before, number: number, width: width, after: after,
                   line: line}
    transform(params)
    return params.before + leftPad(params.number, width, padding) +
           params.after + params.line
  })
  return (isArray ? numbered : numbered.join("\n"))
}

module.exports = lineNumbers
