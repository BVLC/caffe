var aparse = require('acorn').parse;
function parse (src) { return aparse(src, { ecmaVersion: 6 }) }

module.exports = function (src, file) {
    if (typeof src !== 'string') src = String(src);
    
    try {
        eval('throw "STOP"; (function () { ' + src + '})()');
        return;
    }
    catch (err) {
        if (err === 'STOP') return undefined;
        if (err.constructor.name !== 'SyntaxError') throw err;
        return errorInfo(src, file);
    }
};

function errorInfo (src, file) {
    try { parse(src) }
    catch (err) {
        return new ParseError(err, src, file);
    }
    return undefined;
}

function ParseError (err, src, file) {
    SyntaxError.call(this);
    
    this.message = err.message.replace(/\s+\(\d+:\d+\)$/, '');
    
    this.line = err.loc.line;
    this.column = err.loc.column + 1;
    
    this.annotated = '\n'
        + (file || '(anonymous file)')
        + ':' + this.line
        + '\n'
        + src.split('\n')[this.line - 1]
        + '\n'
        + Array(this.column).join(' ') + '^'
        + '\n'
        + 'ParseError: ' + this.message
    ;
}

ParseError.prototype = Object.create(SyntaxError.prototype);

ParseError.prototype.toString = function () {
    return this.annotated;
};

ParseError.prototype.inspect = function () {
    return this.annotated;
};
