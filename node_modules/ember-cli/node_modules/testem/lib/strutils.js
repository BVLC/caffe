// String padding function adapted from <http://jsfromhell.com/string/pad>
function pad(str, l, s, t){
  var ol = l
  return (s || (s = " "), (l -= str.length) > 0 ? (s = new Array(Math.ceil(l / s.length)
    + 1).join(s)).substr(0, t = !t ? l : t == 1 ? 0 : Math.ceil(l / 2))
    + str + s.substr(0, l - t) : str).substring(0, ol)
}

function indent(text, width){
  return text.split('\n').map(function(line){
    return Array((width||4) + 1).join(' ') + line
  }).join('\n')
}

function splitLines(text, colLimit){
  if (!text) return []
  var firstSplit = text.split('\n')
  var secondSplit = []
  firstSplit.forEach(function(line){
    while (line.length > colLimit){
      var first = line.substring(0, colLimit)
      secondSplit.push(first)
      line = line.substring(colLimit)
    }
    secondSplit.push(line)
  })
  return secondSplit
}

// Simple template function. Replaces occurences of "<name>" with param[name]
function template(str, params) {
  return !str.replace ? str : str.replace(/<(.+?)>/g, function(unchanged, name) {
    return name in params ? params[name] : unchanged;
  });
}

exports.pad = pad
exports.indent = indent
exports.splitLines = splitLines
exports.template = template
