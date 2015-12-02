var through = require('through2');

module.exports = function () {
    return through(function (buf, enc, next) {
        this.push(buf.toString('utf8').replace(/X/g, 'Z'));
        next();
    });
};
