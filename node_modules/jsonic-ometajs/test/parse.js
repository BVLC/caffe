var JSONIC = require('../index');
var test = require('tap').test;
var fs = require('fs');

test("Test parse", function(t) {

    fs.readFile(__dirname + '/test.jsonic', {encoding: 'utf-8'}, function (err, contents) {
        if (err) {
            t.fail();
        } else {
            t.deepEqual(JSONIC.parse(contents), {"Hello": "World"}, "parse matches");
        }
        t.end();
    });
});
