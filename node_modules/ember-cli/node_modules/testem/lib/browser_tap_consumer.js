var TapConsumer = require('./tap_consumer')

function BrowserTapConsumer(socket, tapConsumer){
  tapConsumer = tapConsumer || new TapConsumer
  var stream = tapConsumer.stream
  socket.on('tap', function(msg){
    if (!stream.writable) return
    stream.write(msg + '\n')
    if (msg.match(/^#\s+ok\s*$/) || 
      msg.match(/^#\s+fail\s+[0-9]+\s*$/)){
      stream.end()
    }
  })
  return tapConsumer
}

module.exports = BrowserTapConsumer