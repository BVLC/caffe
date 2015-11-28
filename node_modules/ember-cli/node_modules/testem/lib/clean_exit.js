function cleanExit(code) {
  // Workaround for this node core bug <https://github.com/joyent/node/issues/3584>
  // Instead of using `process.exit(?code)`, use this instead.
  //
  var draining = 0;
  var exit = function() {
    if (!(draining--)) {
      process.exit(code);
    }
  };
  var streams = [process.stdout, process.stderr];
  streams.forEach(function(stream) {
    // submit empty write request and wait for completion
    draining += 1;
    stream.write('', exit);
  });
  exit();
}

module.exports = cleanExit
