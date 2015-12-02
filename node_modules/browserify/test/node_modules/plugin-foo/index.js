var through = require('through2');

module.exports = function (b, opts) {
    var stream = through(function () {}, function () {});
    stream.push(opts.msg);
    stream.push(null);
    
    b.pipeline.get('wrap').splice(0,1,stream);
};
