var through = require('through2');

module.exports = function (file) {
    return through(function (buf, enc, next) {
        this.push(String(buf).replace(/GGG/g, '111'));
        next();
    });
};
