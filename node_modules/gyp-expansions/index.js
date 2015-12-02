var ometajs_ = require("ometajs");

var AbstractGrammar = ometajs_.grammars.AbstractGrammar;

var BSJSParser = ometajs_.grammars.BSJSParser;

var BSJSIdentity = ometajs_.grammars.BSJSIdentity;

var BSJSTranslator = ometajs_.grammars.BSJSTranslator;

var cp = require("child_process");

function command(c) {
    return {
        type: "command",
        command: c
    };
}

function str(s) {
    return {
        type: "string",
        value: s
    };
}

function ref(c, v) {
    return {
        type: "ref",
        variable: c,
        value: v[c]
    };
}

function args(a) {
    return {
        type: "arguments",
        value: a
    };
}

var Expansion = function Expansion(source, opts) {
    AbstractGrammar.call(this, source, opts);
};

Expansion.grammarName = "Expansion";

Expansion.match = AbstractGrammar.match;

Expansion.matchAll = AbstractGrammar.matchAll;

exports.Expansion = Expansion;

require("util").inherits(Expansion, AbstractGrammar);

Expansion.prototype["flag"] = function $flag() {
    return this._atomic(function() {
        return this._match("<") && function() {
            return this._options.phase == "pre";
        }.call(this);
    }) || this._atomic(function() {
        return this._match(">") && function() {
            return this._options.phase == "post";
        }.call(this);
    });
};

Expansion.prototype["ref"] = function $ref() {
    return this._atomic(function() {
        var c;
        return this._rule("flag", false, [], null, this["flag"]) && (this._match("@") || this._atomic(function() {
            return this._rule("empty", false, [], null, this["empty"]);
        })) && this._match("(") && this._any(function() {
            return this._atomic(function() {
                return !this._atomic(function() {
                    return this._match(")");
                }, true) && this._rule("char", false, [], null, this["char"]);
            });
        }) && (c = this._getIntermediate(), true) && this._match(")") && this._exec(ref(c.join(""), this._options.variables));
    }) || this._atomic(function() {
        var c;
        return (this._atomic(function() {
            return this._rule("flag", false, [], null, this["flag"]);
        }) || this._atomic(function() {
            return this._rule("empty", false, [], null, this["empty"]);
        })) && this._match("!") && (this._match("@") || this._atomic(function() {
            return this._rule("empty", false, [], null, this["empty"]);
        })) && this._match("(") && this._rule("command", false, [], null, this["command"]) && (c = this._getIntermediate(), 
        true) && this._match(")") && this._exec(command(c));
    });
};

Expansion.prototype["command"] = function $command() {
    return this._atomic(function() {
        var a;
        return this._match("[") && this._rule("listOf", false, [ "string", "," ], null, this["listOf"]) && (a = this._getIntermediate(), 
        true) && this._match("]") && this._exec(args(a));
    }) || this._atomic(function() {
        var s;
        return this._any(function() {
            return this._atomic(function() {
                return this._rule("stringInside", false, [ ")" ], null, this["stringInside"]);
            });
        }) && (s = this._getIntermediate(), true) && this._exec(s);
    });
};

Expansion.prototype["string"] = function $string() {
    return this._atomic(function() {
        var c;
        return this._match('"') && this._any(function() {
            return this._atomic(function() {
                return this._rule("stringInside", false, [ '"' ], null, this["stringInside"]);
            });
        }) && (c = this._getIntermediate(), true) && this._match('"') && this._exec(c);
    }) || this._atomic(function() {
        var c;
        return this._match("'") && this._any(function() {
            return this._atomic(function() {
                return this._rule("stringInside", false, [ "'" ], null, this["stringInside"]);
            });
        }) && (c = this._getIntermediate(), true) && this._match("'") && this._exec(c);
    });
};

Expansion.prototype["stringInside"] = function $stringInside() {
    var x;
    return this._skip() && (x = this._getIntermediate(), true) && (this._atomic(function() {
        var r;
        return this._rule("ref", false, [], null, this["ref"]) && (r = this._getIntermediate(), 
        true) && this._exec(r);
    }) || this._atomic(function() {
        var p;
        return this._many(function() {
            return this._atomic(function() {
                return !this._atomic(function() {
                    return this._rule("ref", false, [], null, this["ref"]);
                }, true) && this._rule("stringInsideParts", false, [ x ], null, this["stringInsideParts"]);
            });
        }) && (p = this._getIntermediate(), true) && this._exec(str(p.join("")));
    }));
};

Expansion.prototype["stringInsideRaw"] = function $stringInsideRaw() {
    return this._atomic(function() {
        var r;
        return this._rule("ref", false, [], null, this["ref"]) && (r = this._getIntermediate(), 
        true) && this._exec(r);
    }) || this._atomic(function() {
        var p;
        return this._many(function() {
            return this._atomic(function() {
                return !this._atomic(function() {
                    return this._rule("ref", false, [], null, this["ref"]);
                }, true) && this._rule("stringInsidePartsRaw", false, [], null, this["stringInsidePartsRaw"]);
            });
        }) && (p = this._getIntermediate(), true) && this._exec(str(p.join("")));
    });
};

Expansion.prototype["stringInsidePartsRaw"] = function $stringInsidePartsRaw() {
    return this._atomic(function() {
        var y;
        return this._match("\\") && this._rule("char", false, [], null, this["char"]) && (y = this._getIntermediate(), 
        true) && this._exec(y);
    }) || this._atomic(function() {
        var c;
        return this._rule("char", false, [], null, this["char"]) && (c = this._getIntermediate(), 
        true) && this._exec(c);
    });
};

Expansion.prototype["stringInsideParts"] = function $stringInsideParts() {
    var x;
    return this._skip() && (x = this._getIntermediate(), true) && (this._atomic(function() {
        var y;
        return this._match("\\") && this._rule("char", false, [], null, this["char"]) && (y = this._getIntermediate(), 
        true) && this._exec(y);
    }) || this._atomic(function() {
        var c;
        return this._atomic(function() {
            return !this._atomic(function() {
                return this._rule("seq", false, [ x ], null, this["seq"]);
            }, true) && this._rule("char", false, [], null, this["char"]);
        }) && (c = this._getIntermediate(), true) && this._exec(c);
    }));
};

Expansion.prototype["stringRaw"] = function $stringRaw() {
    var c;
    return this._any(function() {
        return this._atomic(function() {
            return this._rule("stringInsideRaw", false, [], null, this["stringInsideRaw"]);
        });
    }) && (c = this._getIntermediate(), true) && this._exec(c);
};

function iter(array, proc, cb) {
    var done = 0;
    var result = [];
    for (var i = 0; i < array.length; i++) {
        (function(j) {
            setImmediate(function() {
                proc(array[j], function(err, res) {
                    if (err) cb(err);
                    result[j] = res;
                    if (++done == array.length) cb(null, result);
                });
            });
        })(i);
    }
}

function handleNode(e, cb) {
    if (e.type == "command") {
        if (Array.isArray(e.command)) {
            iter(e.command, handleNode, function(err, res) {
                if (err) return cb(err);
                cp.exec(res.join(""), function(err, stdout) {
                    if (err) return cb(err);
                    cb(null, stdout[stdout.length - 1] == "\n" ? stdout.slice(0, stdout.length - 1) : stdout);
                });
            });
        } else if (e.command.type == "arguments") {
            iter(e.command.value, function(n, cb) {
                iter(n, handleNode, function(err, res) {
                    if (err) return cb(err);
                    cb(null, res.join(""));
                });
            }, function(err, res) {
                cp.execFile(res[0], res.slice(1), function(err, stdout) {
                    if (err) return cb(err);
                    cb(null, stdout[stdout.length - 1] == "\n" ? stdout.slice(0, stdout.length - 1) : stdout);
                });
            });
        } else {
            cb(new Error("unknown command node type"));
        }
    } else if (e.type == "ref") {
        cb(null, e.value);
    } else if (e.type == "string") {
        cb(null, e.value);
    } else {
        cb(null, e);
    }
}

var expansions = module.exports = {
    parser: Expansion,
    expandString: function(s, variables, which, cb) {
        var tree = Expansion.matchAll(s, "stringRaw", {
            variables: variables,
            phase: which
        });
        iter(tree, handleNode, function(err, res) {
            if (err) return cb(err);
            setImmediate(function() {
                cb(null, res.join(""));
            });
        });
    },
    expandArray: function(a, variables, which, cb) {
        var re = which == "pre" ? /^<@\((.*)\)$/ : /^>@\((.*)\)$/;
        var out = [];
        var m;
        for (var i = 0; i < a.length; i++) {
            if (m = re.exec(a[i])) {
                if (Array.isArray(variables[m[1]])) {
                    variables[m[1]].forEach(function(e) {
                        out.push(e);
                    });
                } else {
                    out.push(variables[m[1]]);
                }
            } else {
                out.push(a[i]);
            }
        }
        cb(null, out);
    }
};