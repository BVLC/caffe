var merge = require('gyp-merge');
var path = require('path');
var fs = require('fs');
var expansions = require('gyp-expansions');
var load = require('gyp-load');
var cond = require('gyp-conditions');

var gyp = module.exports = function gyp(arg, variables, cb) {
    if (typeof arg == 'string' || arg instanceof String) {
        arg = load(arg);
    }

    if (typeof arg != 'object') {
        throw new Error("Root must be an object");
    }

    handle(arg, variables, 'pre', cb);

    return;

    /// @todo do post-phase expansions

    cb(null, arg);
};

function recurse(obj, variables, which, cb) {
    oiter(obj, function(e, c) {
        handle(e, variables, which, c);
    }, cb);
}

function oiter(obj, proc, cb) {
    var done = 0;
    var result = {};
    var length = Object.keys(obj).length;
    for (var i in obj) {
        (function(j) {
            setImmediate(function() {
                proc(obj[j], function(err, res) {
                    if (err == 'continue') {
                        if (++done == length) cb(null, result);
                    } else if (err) {
                        cb(err);
                    } else {
                        result[j] = res;
                        if (++done == length) cb(null, result);
                    }
                }, j);
            });
        })(i);
    }
}

function aiter(obj, proc, cb) {
    var done = 0;
    var result = {};
    var length = obj.length;
    for (var i = 0; i < length; i++) {
        (function(j) {
            setImmediate(function() {
                proc(obj[j], function(err, res) {
                    if (err == 'continue') {
                        if (++done == length) cb(null, result);
                    } else if (err) {
                        cb(err);
                    } else {
                        result[j] = res;
                        if (++done == length) cb(null, result);
                    }
                }, j);
            });
        })(i);
    }
}

function handle(thing, variables, which, cb) {
    if (Array.isArray(thing)) {
        expansions.expandArray(thing, variables, which, function (err, res) {
            if (err) return cb(err);

            aiter(typeof res != 'undefined' ? res : thing, function (e, cont) {
                handle(e, variables, which, cont);
            }, cb);
        });
    } else if (typeof thing == 'object') {
        var m;
        var out = {};
        if (thing.variables) {
            variables = Object.create(variables);
            for (var v in thing.variables) {
                if (m = /(.*)%$/.exec(v) && !variables[m[1]]) {
                    variables[m[1]] = thing.variables[v];
                } else {
                    variables[v] = thing.variables[v];
                }
            }
        }

        if (thing.conditions) {
            thing.conditions.forEach(function (e) {
                if (cond(e[0], variables)) {
                    thing = merge(e[1], thing);
                } else if (e[2]) {
                    thing = merge(e[2], thing);
                }
            });
        }

        oiter(thing, function(e, cont, key) {
            if (key == 'variables') return cont('continue');
            handle(e, variables, which, cont);
        }, cb);
    } else {
        thing = "" + thing;
        expansions.expandString(thing, variables, which, function(err, res) {
            cb(null, typeof res != 'undefined' ? res : thing);
        });
    }
}
