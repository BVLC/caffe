var test = require('tap').test;
var resolve = require('../');

test('mock', function (t) {
    t.plan(4);
    
    var files = {
        '/foo/bar/baz.js' : 'beep'
    };
    
    function opts (basedir) {
        return {
            basedir : basedir,
            isFile : function (file, cb) {
                cb(null, files.hasOwnProperty(file));
            },
            readFile : function (file, cb) {
                cb(null, files[file]);
            }
        }
    }
    
    resolve('./baz', opts('/foo/bar'), function (err, res) {
        if (err) t.fail(err);
        t.equal(res, '/foo/bar/baz.js');
    });
    
    resolve('./baz.js', opts('/foo/bar'), function (err, res) {
        if (err) t.fail(err);
        t.equal(res, '/foo/bar/baz.js');
    });
    
    resolve('baz', opts('/foo/bar'), function (err, res) {
        t.equal(err.message, "Cannot find module 'baz'");
    });
    
    resolve('../baz', opts('/foo/bar'), function (err, res) {
        t.equal(err.message, "Cannot find module '../baz'");
    });
});

test('mock package', function (t) {
    t.plan(1);
    
    var files = {
        '/foo/node_modules/bar/baz.js' : 'beep',
        '/foo/node_modules/bar/package.json' : JSON.stringify({
            main : './baz.js'
        })
    };
    
    function opts (basedir) {
        return {
            basedir : basedir,
            isFile : function (file, cb) {
                cb(null, files.hasOwnProperty(file));
            },
            readFile : function (file, cb) {
                cb(null, files[file]);
            }
        }
    }
    
    resolve('bar', opts('/foo'), function (err, res) {
        if (err) t.fail(err);
        t.equal(res, '/foo/node_modules/bar/baz.js');
    });
});
