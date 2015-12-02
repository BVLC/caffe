var path = require('path'),
    fs = require('./fs'),
    PromiseConstructor,
    AbstractFileManager = require("../less/environment/abstract-file-manager.js");

try {
    PromiseConstructor = typeof Promise === 'undefined' ? require('promise') : Promise;
} catch(e) {
}

var FileManager = function() {
};

FileManager.prototype = new AbstractFileManager();

FileManager.prototype.supports = function(filename, currentDirectory, options, environment) {
    return true;
};
FileManager.prototype.supportsSync = function(filename, currentDirectory, options, environment) {
    return true;
};

FileManager.prototype.loadFile = function(filename, currentDirectory, options, environment, callback) {
    var fullFilename,
        data,
        isAbsoluteFilename = this.isPathAbsolute(filename),
        filenamesTried = [];

    options = options || {};

    if (options.syncImport || !PromiseConstructor) {
        data = this.loadFileSync(filename, currentDirectory, options, environment, 'utf-8');
        callback(data.error, data);
        return;
    }

    var paths = isAbsoluteFilename ? [""] : [currentDirectory];
    if (options.paths) { paths.push.apply(paths, options.paths); }
    if (!isAbsoluteFilename && paths.indexOf('.') === -1) { paths.push('.'); }

    // promise is guarenteed to be asyncronous
    // which helps as it allows the file handle
    // to be closed before it continues with the next file
    return new PromiseConstructor(function(fulfill, reject) {
        (function tryPathIndex(i) {
            if (i < paths.length) {
                fullFilename = filename;
                if (paths[i]) {
                    fullFilename = path.join(paths[i], fullFilename);
                }
                fs.stat(fullFilename, function (err) {
                    if (err) {
                        filenamesTried.push(fullFilename);
                        tryPathIndex(i + 1);
                    } else {
                        fs.readFile(fullFilename, 'utf-8', function(e, data) {
                            if (e) { reject(e); return; }

                            fulfill({ contents: data, filename: fullFilename});
                        });
                    }
                });
            } else {
                reject({ type: 'File', message: "'" + filename + "' wasn't found. Tried - " + filenamesTried.join(",") });
            }
        }(0));
    });
};

FileManager.prototype.loadFileSync = function(filename, currentDirectory, options, environment, encoding) {
    var fullFilename, paths, filenamesTried = [], isAbsoluteFilename = this.isPathAbsolute(filename) , data;
    options = options || {};

    paths = isAbsoluteFilename ? [""] : [currentDirectory];
    if (options.paths) {
        paths.push.apply(paths, options.paths);
    }
    if (!isAbsoluteFilename && paths.indexOf('.') === -1) {
        paths.push('.');
    }

    var err, result;
    for (var i = 0; i < paths.length; i++) {
        try {
            fullFilename = filename;
            if (paths[i]) {
                fullFilename = path.join(paths[i], fullFilename);
            }
            filenamesTried.push(fullFilename);
            fs.statSync(fullFilename);
            break;
        } catch (e) {
            fullFilename = null;
        }
    }

    if (!fullFilename) {
        err = { type: 'File', message: "'" + filename + "' wasn't found. Tried - " + filenamesTried.join(",") };
        result = { error: err };
    } else {
        data = fs.readFileSync(fullFilename, encoding);
        result = { contents: data, filename: fullFilename};
    }

    return result;
};

module.exports = FileManager;
