var test = require('tap').test;
var JSON = require('../index');
var fs = require('fs');
var reference = require('./test.json');

test('full parse', function (t) {
    fs.readFile(__dirname + '/test.json', { encoding: 'utf-8' }, function (err, contents) {
        if (err) {
            t.fail(err);
        } else {
            t.deepEqual(JSON.parse(contents), reference, "parse matches native parser");
        }
        t.end();
    });
});
