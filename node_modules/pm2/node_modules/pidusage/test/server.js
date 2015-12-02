var http = require('http'), pusage = require('../')

http.createServer(function(req, res) {
  res.writeHead(200)
  res.end("hello world\n")
}).listen(8020)

var interval = setInterval(function () {
  console.log('\033[2J')
  pusage.stat(process.pid, function(err, stat) {
    console.log(stat)
  })
}, 100)

process.on('exit', function() {
    clearInterval(interval)
})
