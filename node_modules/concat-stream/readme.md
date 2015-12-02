# concat-stream

    npm install concat-stream

then

    var concat = require('concat-stream')
    var fs = require('fs')
    
    var read = fs.createReadStream('readme.md')
    var write = concat(function(err, data) {})
    
    read.pipe(write)
    
works with arrays too!

    var write = concat(function(err, data) {})
    write.write([1,2,3])
    write.write([4,5,6])
    write.end()
    // data will be [1,2,3,4,5,6] in the above callback

works with buffers too! can't believe the deals!

    var write = concat(function(err, data) {})
    write.write(new Buffer('hello '))
    write.write(new Buffer('world'))
    write.end()
    // data will be a buffer that toString()s to 'hello world' in the above callback
    
MIT LICENSE
