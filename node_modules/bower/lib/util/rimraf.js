var rimraf = require('rimraf');
var chmodr = require('chmodr');
var fs = require('./fs');

module.exports = function (dir, callback) {
    var checkAndRetry = function (e) {
        fs.lstat(dir, function (err, stats) {
            if (err) {
                if (err.code === 'ENOENT') return callback();
                return callback(e);
            }

            chmodr(dir, 0777, function (err) {
                if (err) return callback(e);
                rimraf(dir, callback);
            });
        });
    };

    if (process.platform === 'win32') {
        checkAndRetry();
    } else {
        rimraf(dir, checkAndRetry);
    }
};

module.exports.sync = function (dir) {
    var checkAndRetry = function () {
        try {
            fs.lstatSync(dir);
            chmodr.sync(dir, 0777);
            return rimraf.sync(dir);
        } catch (e) {
            if (e.code === 'ENOENT') return;
            throw e;
        }
    };

    try {
        return rimraf.sync(dir);
    } catch (e) {
        return checkAndRetry();
    } finally {
        return checkAndRetry();
    }
};
