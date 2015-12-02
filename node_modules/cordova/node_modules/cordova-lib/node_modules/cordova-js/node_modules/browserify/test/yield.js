var browserify = require('../');
var test = require('tap').test;
var vm = require('vm');

test('yield', function (t) {
    t.plan(6);
    var b = browserify(__dirname + '/yield/main.js');
    b.transform('es6ify');
    
    b.bundle(function (err, src) {
        t.error(err);
        var c = { console: { log: log } };
        var index = 0;
        vm.runInNewContext(src, c);
        
        function log (msg) {
            t.equal(index++, msg);
        }
    });
});
