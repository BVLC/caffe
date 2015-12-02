var merge = require('gyp-merge');
var path = require('path');
var fs = require('fs');
var JSONIC = require('jsonic-ometajs');

var gyp = module.exports = function gyp(arg, relative) {
    arg = loadFile(arg); /// @todo replace with a parser that supports comments.

    if (typeof arg != 'object') {
        throw new Error("Root must be an object");
    }

    return parse(arg);

    function parse(arg) {
        if (Array.isArray(arg)) {
            return arg.map(parse);
        } else if (typeof arg == 'string' || arg instanceof String) {
            return arg; /// @todo do string interpolations
        } else if (typeof arg == 'number') {
            return arg;
        } else if (typeof arg == 'object') {
            return parseObject(arg);
        } else {
            throw new Error("GYP objects must consist of lists, dictionaries, and scalar values only, got " + typeof arg);
        }
    }

    function parseObject(arg) {
        var out = {};
        if (arg.includes) {
            if (!Array.isArray(arg.includes)) {
                throw new Error("includes found but not an array");
            }

            arg.includes.forEach(function (e) {
                arg = merge(arg, gyp(e, relative));
            });
        }

        for (var k in arg) {
            if (k == 'includes') continue;
            out[k] = parse(arg[k]);
        }

        return out;
    }

    function loadFile(filename) {
        return JSONIC.parse(fs.readFileSync(relative ? path.resolve(relative, filename) : filename, {encoding: 'utf-8'}));
    }
};
