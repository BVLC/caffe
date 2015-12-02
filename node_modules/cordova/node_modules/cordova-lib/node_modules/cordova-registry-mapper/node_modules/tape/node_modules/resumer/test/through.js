var test = require('tape');
var resumer = require('../');
var concat = require('concat-stream');

test('through write/end', function (t) {
    t.plan(2);
    
    var s = createStream();
    
    s.on('okok', function () {
        t.ok(true);
    });
    
    s.pipe(concat(function (err, body) {
        t.equal(body, 'BEGIN\nRAWR\nEND\n');
    }));
    
    setTimeout(function () {
        s.end('rawr\n');
    }, 50);
});

function createStream () {
    var stream = resumer(write, end);
    stream.queue('BEGIN\n');
    return stream;
    
    function write (x) {
        this.queue(String(x).toUpperCase());
    }
    function end () {
        this.emit('okok');
        this.queue('END\n');
        this.queue(null);
    }
}
