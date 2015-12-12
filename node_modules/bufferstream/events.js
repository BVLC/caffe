fs = require('fs')
http = require('http')
util = require('util')
Stream = require('stream').Stream
spawn = require('child_process').spawn

filename = "/home/dodo/Videos/Primer.avi"


ReadableAndWritableStream = function () {
    Stream.call(this)
    this.paused = false
    this.readable = true
    this.writable = true
}
util.inherits(ReadableAndWritableStream, Stream)

ReadableAndWritableStream.prototype.write = function (data) {
//     console.log("recv data", data.length)
    var p1 = this.paused
    if (!this.paused) this.emit('data', data)
    var p2 = this.paused
    if (p1 !== p2) {
        console.log(p1, p2)
        process.exit(1)
    }
    return !this.paused
}

ReadableAndWritableStream.prototype.pause = function (data) {
    console.log("paused")
    this.paused = true
}

ReadableAndWritableStream.prototype.resume = function (data) {
    console.log("resumed")
    this.emit('drain')
    this.paused = false
}

ReadableAndWritableStream.prototype.end = function (data) {
    console.log("ended")
    this.emit('end')
    this.emit('close')
}



fs.stat(filename, function (err, stat) {
    if (err) throw err
    var srv = http.createServer(function (req, res) {
        console.log("request")
        var file = spawn('cat', [filename])

        res.setHeader('Content-Length', stat.size)

        res.on('pause',  console.log.bind(console, "res pause"))
        res.on('resume', console.log.bind(console, "res resume"))
        res.on('drain',  console.log.bind(console, "res drain"))
        res.on('end',    console.log.bind(console, "res end"))
        res.on('error',  console.log.bind(console, "res error"))
//         res.on('data',   console.log.bind(console, "res data"))

        file.stdout.on('pause',  console.log.bind(console, "file pause"))
        file.stdout.on('resume', console.log.bind(console, "file resume"))
        file.stdout.on('drain',  console.log.bind(console, "file drain"))
//         file.stdout.on('data',   console.log.bind(console, "file data"))
        file.stdout.on('end',    console.log.bind(console, "file end"))
        file.on('exit', console.log.bind(console, "file exit"))

//         return file.stdout.pipe(res) // works

        var proxy = new ReadableAndWritableStream
        file.stdout.pipe(proxy).pipe(res) // doesnt work :(
        // 0.4.x compatibility
//         proxy.pipe(res)
//         file.stdout.pipe(proxy)
        // but doesnt work either

    }).listen(3000, 'localhost')
    console.log("listen on localhost:3000")
})
