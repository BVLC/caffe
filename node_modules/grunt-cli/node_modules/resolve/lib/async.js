var core = require('./core');
var fs = require('fs');
var path = require('path');

module.exports = function resolve (x, opts, cb) {
    if (core[x]) return cb(null, x);
    
    if (typeof opts === 'function') {
        cb = opts;
        opts = {};
    }
    if (!opts) opts = {};
    
    var isFile = opts.isFile || function (file, cb) {
        fs.stat(file, function (err, stat) {
            if (err && err.code === 'ENOENT') cb(null, false)
            else if (err) cb(err)
            else cb(null, stat.isFile() || stat.isFIFO())
        });
    };
    var readFile = opts.readFile || fs.readFile;
    
    var extensions = opts.extensions || [ '.js' ];
    var y = opts.basedir
        || path.dirname(require.cache[__filename].parent.filename)
    ;
    
    opts.paths = opts.paths || [];
    
    if (x.match(/^(?:\.\.?\/|\/|([A-Za-z]:)?\\)/)) {
        loadAsFile(path.resolve(y, x), function (err, m) {
            if (err) cb(err)
            else if (m) cb(null, m)
            else loadAsDirectory(path.resolve(y, x), function (err, d) {
                if (err) cb(err)
                else if (d) cb(null, d)
                else cb(new Error("Cannot find module '" + x + "'"))
            })
        });
    }
    else loadNodeModules(x, y, function (err, n) {
        if (err) cb(err)
        else if (n) cb(null, n)
        else cb(new Error("Cannot find module '" + x + "'"))
    });
    
    function loadAsFile (x, cb) {
        (function load (exts) {
            if (exts.length === 0) return cb(null, undefined);
            var file = x + exts[0];
            
            isFile(file, function (err, ex) {
                if (err) cb(err)
                else if (ex) cb(null, file)
                else load(exts.slice(1))
            });
        })([''].concat(extensions));
    }
    
    function loadAsDirectory (x, cb) {
        var pkgfile = path.join(x, '/package.json');
        isFile(pkgfile, function (err, ex) {
            if (err) return cb(err);
            if (!ex) return loadAsFile(path.join(x, '/index'), cb);
            
            readFile(pkgfile, function (err, body) {
                if (err) return cb(err);
                try {
                    var pkg = JSON.parse(body);
                }
                catch (err) {}
                
                if (opts.packageFilter) {
                    pkg = opts.packageFilter(pkg, x);
                }
                
                if (pkg.main) {
                    loadAsFile(path.resolve(x, pkg.main), function (err, m) {
                        if (err) return cb(err);
                        if (m) return cb(null, m);
                        var dir = path.resolve(x, pkg.main);
                        loadAsDirectory(dir, function (err, n) {
                            if (err) return cb(err);
                            if (n) return cb(null, n);
                            loadAsFile(path.join(x, '/index'), cb);
                        });
                    });
                    return;
                }
                
                loadAsFile(path.join(x, '/index'), cb);
            });
        });
    }
    
    function loadNodeModules (x, start, cb) {
        (function process (dirs) {
            if (dirs.length === 0) return cb(null, undefined);
            var dir = dirs[0];
            
            loadAsFile(path.join(dir, '/', x), function (err, m) {
                if (err) return cb(err);
                if (m) return cb(null, m);
                loadAsDirectory(path.join(dir, '/', x), function (err, n) {
                    if (err) return cb(err);
                    if (n) return cb(null, n);
                    process(dirs.slice(1));
                });
            });
        })(nodeModulesPaths(start));
    }
    
    function nodeModulesPaths (start, cb) {
        var splitRe = process.platform === 'win32' ? /[\/\\]/ : /\/+/;
        var parts = start.split(splitRe);
        
        var dirs = [];
        for (var i = parts.length - 1; i >= 0; i--) {
            if (parts[i] === 'node_modules') continue;
            var dir = path.join(
                path.join.apply(path, parts.slice(0, i + 1)),
                'node_modules'
            );
            if (!parts[0].match(/([A-Za-z]:)/)) {
                dir = '/' + dir;    
            }
            dirs.push(dir);
        }
        return dirs.concat(opts.paths);
    }
};
