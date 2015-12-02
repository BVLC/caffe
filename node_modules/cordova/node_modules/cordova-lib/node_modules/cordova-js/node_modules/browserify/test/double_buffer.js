var browserify = require('../');
var test = require('tap').test;
var vm = require('vm');

test('double buffer', function (t) {
    t.plan(1);
    
    var b = browserify(__dirname + '/double_buffer/main.js');
    b.require('buffer');
    b.bundle(function (err, src) {
        if (err) return t.fail(err);
        vm.runInNewContext(src, { t: t });
    });
});
