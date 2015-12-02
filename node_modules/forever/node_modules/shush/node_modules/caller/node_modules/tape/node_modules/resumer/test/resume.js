var test = require('tape');
var resumer = require('../');
var concat = require('concat-stream');

test('implicit resume', function (t) {
    t.plan(1);
    
    var s = createStream();
    s.pipe(concat(function (err, body) {
        t.equal(body, 'beep boop\n');
    }));
});

test('pause/resume', function (t) {
    t.plan(2);
    
    var s = createStream();
    s.pause();
    
    var paused = true;
    setTimeout(function () {
        paused = false;
        s.resume();
    }, 100);
    
    s.pipe(concat(function (err, body) {
        t.equal(paused, false);
        t.equal(body, 'beep boop\n');
    }));
});

function createStream () {
    var stream = resumer();
    stream.queue('beep boop\n');
    stream.queue(null);
    return stream;
}
