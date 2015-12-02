'use strict';

var fs = require('fs'),
    path = require('path'),
    caller = require('caller'),
    strip = require('strip-json-comments');


function tryParse(file, json) {
    var err;
    try {
        return JSON.parse(json);
    } catch (e) {
        err = new Error(e.message  + ' in ' + file);
        err.__proto__ = e.__proto__;
        throw err;
    }
}

function shush(file) {
    var root, abs, json;

    root = path.resolve(caller());
    root = path.dirname(root);

    abs = path.resolve(root, file);
    abs = require.resolve(abs);

    json = fs.readFileSync(abs, 'utf8');
    json = strip(json);

    return tryParse(abs, json);
}


//shush.createStream = function () {
//
//};


module.exports = shush;