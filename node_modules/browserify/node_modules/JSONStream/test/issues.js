var JSONStream = require('../');
var test = require('tape')

test('#66', function (t) {
   var error = 0;
   var stream = JSONStream
    .parse()
    .on('error', function (err) {
        t.ok(err);
        error++;
    })
    .on('end', function () {
        t.ok(error === 1);
        t.end();
    });

    stream.write('["foo":bar[');
    stream.end();

});
