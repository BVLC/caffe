var test = require('tap').test;
var resolve = require('../');

test('filter', function (t) {
    t.plan(1);
    var dir = __dirname + '/resolver';
    resolve('./baz', {
        basedir : dir,
        packageFilter : function (pkg) {
            pkg.main = 'doom';
            return pkg;
        }
    }, function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/baz/doom.js');
    });
});
