var resumer = require('../');
createStream().pipe(process.stdout);

function createStream () {
    var stream = resumer();
    stream.queue('beep boop\n');
    return stream;
}
