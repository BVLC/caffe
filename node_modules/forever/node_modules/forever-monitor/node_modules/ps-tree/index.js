var spawn = require('child_process').spawn, 
    es    = require('event-stream');

module.exports = childrenOfPid

function childrenOfPid( pid, callback) {
  var headers = null;

  if('function' !== typeof callback) 
    throw new Error('childrenOfPid(pid, callback) expects callback')
  if('number' == typeof pid)
    pid = pid.toString()
  
  es.connect(
    spawn('ps', ['-A', '-o', 'ppid,pid']).stdout,
    es.split(),
    es.map(function (line, cb) { //this could parse alot of unix command output
      var columns = line.trim().split(/\s+/)
      if(!headers)
        headers = columns
      else {
        var row = {}
        //for each header, 
        var h = headers.slice()
        while (h.length) {
          row[h.shift()] = h.length ? columns.shift() : columns.join(' ')
        }
        return cb(null, row)
      }
      return cb()
    }),
    es.writeArray(function (err, ps) {
      var parents = [pid], children = []
      ps.forEach(function (proc) {
        if(-1 != parents.indexOf(proc.PPID)) {
          parents.push(proc.PID)
          children.push(proc)
        }
      })
      callback(null, children)    
    })
  ).on('error', callback)
}

if(!module.parent) {
  childrenOfPid(process.argv[2] || 1, function (err, data) {
  console.log(data)
  })
}