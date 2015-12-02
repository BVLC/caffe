var test = require('tap').test;
var resolve = require('../');

test('async foo', function (t) {
    t.plan(3);
    var dir = __dirname + '/resolver';
    
    resolve('./foo', { basedir : dir }, function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/foo.js');
    });
    
    resolve('./foo.js', { basedir : dir }, function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/foo.js');
    });
    
    resolve('foo', { basedir : dir }, function (err) {
        t.equal(err.message, "Cannot find module 'foo'");
    });
});

test('bar', function (t) {
    t.plan(2);
    var dir = __dirname + '/resolver';
    
    resolve('foo', { basedir : dir + '/bar' }, function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/bar/node_modules/foo/index.js');
    });
    
    resolve('foo', { basedir : dir + '/bar' }, function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/bar/node_modules/foo/index.js');
    });
});

test('baz', function (t) {
    t.plan(1);
    var dir = __dirname + '/resolver';
    
    resolve('./baz', { basedir : dir }, function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/baz/quux.js');
    });
});

test('biz', function (t) {
    t.plan(3);
    var dir = __dirname + '/resolver/biz/node_modules';
    
    resolve('./grux', { basedir : dir }, function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/grux/index.js');
    });
    
    resolve('tiv', { basedir : dir + '/grux' }, function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/tiv/index.js');
    });
    
    resolve('grux', { basedir : dir + '/tiv' }, function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/grux/index.js');
    });
});

test('normalize', function (t) {
    t.plan(1);
    var dir = __dirname + '/resolver/biz/node_modules/grux';
    
    resolve('../grux', { basedir : dir }, function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/index.js');
    });
});

test('cup', function (t) {
    t.plan(3);
    var dir = __dirname + '/resolver';
    
    resolve('./cup', { basedir : dir, extensions : [ '.js', '.coffee' ] },
    function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/cup.coffee');
    });
    
    resolve('./cup.coffee', { basedir : dir }, function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/cup.coffee');
    });
    
    resolve('./cup', { basedir : dir, extensions : [ '.js' ] },
    function (err, res) {
        t.equal(err.message, "Cannot find module './cup'");
    });
});

test('mug', function (t) {
    t.plan(3);
    var dir = __dirname + '/resolver';
    
    resolve('./mug', { basedir : dir }, function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/mug.js');
    });
    
    resolve('./mug', { basedir : dir, extensions : [ '.coffee', '.js' ] },
    function (err, res) {
        if (err) t.fail(err);
        t.equal(res, dir + '/mug.coffee');
    });
    
    resolve('./mug', { basedir : dir, extensions : [ '.js', '.coffee' ] },
    function (err, res) {
        t.equal(res, dir + '/mug.js');
    });
});

test('other path', function (t) {
    t.plan(4);
    var resolverDir = __dirname + '/resolver';
    var dir = resolverDir + '/bar';
    var otherDir = resolverDir + '/other_path';
    
    resolve('root', { basedir : dir, paths: [otherDir] }, function (err, res) {
        if (err) t.fail(err);
        t.equal(res, resolverDir + '/other_path/root.js');
    });
    
    resolve('lib/other-lib', { basedir : dir, paths: [otherDir] },
    function (err, res) {
        if (err) t.fail(err);
        t.equal(res, resolverDir + '/other_path/lib/other-lib.js');
    });
    
    resolve('root', { basedir : dir, }, function (err, res) {
        t.equal(err.message, "Cannot find module 'root'");
    });
    
    resolve('zzz', { basedir : dir, paths: [otherDir] }, function (err, res) {
        t.equal(err.message, "Cannot find module 'zzz'");
    });
});
