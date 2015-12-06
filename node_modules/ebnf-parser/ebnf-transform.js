var EBNF = (function(){
    var parser = require('./transform-parser.js');

    var transformExpression = function(e, opts, emit) {
        var type = e[0], value = e[1], name = false;

        if (type === 'xalias') {
            type = e[1];
            value = e[2]
            name = e[3];
            if (type) {
                e = e.slice(1,2);
            } else {
                e = value;
                type = e[0];
                value = e[1];
            }
        }

        if (type === 'symbol') {
            var n;
            if (e[1][0] === '\\') n = e[1][1];
            else if (e[1][0] === '\'') n = e[1].substring(1, e[1].length-1);
            else n = e[1];
            emit(n + (name ? "["+name+"]" : ""));
        } else if (type === "+") {
            if (!name) {
                name = opts.production + "_repetition_plus" + opts.repid++;
            }
            emit(name);

            opts = optsForProduction(name, opts.grammar);
            var list = transformExpressionList([value], opts);
            opts.grammar[name] = [
                [list, "$$ = [$1];"],
                [
                    name + " " + list,
                    "$1.push($2);"
                ]
            ];
        } else if (type === "*") {
            if (!name) {
                name = opts.production + "_repetition" + opts.repid++;
            }
            emit(name);

            opts = optsForProduction(name, opts.grammar);
            opts.grammar[name] = [
                ["", "$$ = [];"],
                [
                    name + " " + transformExpressionList([value], opts),
                    "$1.push($2);"
                ]
            ];
        } else if (type ==="?") {
            if (!name) {
                name = opts.production + "_option" + opts.optid++;
            }
            emit(name);

            opts = optsForProduction(name, opts.grammar);
            opts.grammar[name] = [
                "", transformExpressionList([value], opts)
            ];
        } else if (type === "()") {
            if (value.length == 1) {
                emit(transformExpressionList(value[0], opts));
            } else {
                if (!name) {
                    name = opts.production + "_group" + opts.groupid++;
                }
                emit(name);

                opts = optsForProduction(name, opts.grammar);
                opts.grammar[name] = value.map(function(handle) {
                    return transformExpressionList(handle, opts);
                });
            }
        }
    };

    var transformExpressionList = function(list, opts) {
        return list.reduce (function (tot, e) {
            transformExpression (e, opts, function (i) { tot.push(i); });
            return tot;
        }, []).
        join(" ");
    };

    var optsForProduction = function(id, grammar) {
        return {
            production: id,
            repid: 0,
            groupid: 0,
            optid: 0,
            grammar: grammar
        };
    };

    var transformProduction = function(id, production, grammar) {
        var transform_opts = optsForProduction(id, grammar);
        return production.map(function (handle) {
            var action = null, opts = null;
            if (typeof(handle) !== 'string')
                action = handle[1],
                opts = handle[2],
                handle = handle[0];
            var expressions = parser.parse(handle);

            handle = transformExpressionList(expressions, transform_opts);

            var ret = [handle];
            if (action) ret.push(action);
            if (opts) ret.push(opts);
            if (ret.length == 1) return ret[0];
            else return ret;
        });
    };

    var transformGrammar = function(grammar) {
        Object.keys(grammar).forEach(function(id) {
            grammar[id] = transformProduction(id, grammar[id], grammar);
        });
    };

    return {
        transform: function (ebnf) {
            transformGrammar(ebnf);
            return ebnf;
        }
    };
})();

exports.transform = EBNF.transform;

