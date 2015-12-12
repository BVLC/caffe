if (typeof require === "function") var Parser = require("jison").Parser;

var parsexp = new Parser({
    "bnf": {
        "pattern": [
            ["expression EOF", "return $1;"]
        ],
        "expression": [
            ["expression | sequence", "$$ = yy.expression($1, $3);"],
            ["sequence",              "$$ = $1;"]
        ],
        "sequence": [
            ["sequence group", "$$ = yy.sequence($1, $2);"],
            ["group",          "$$ = $1;"]
        ],
        "group": [
            ["atom *", "$$ = yy.group($1);"],
            ["atom",   "$$ = $1;"]
        ],
        "atom": [
            ["SYMBOL",         "$$ = yy.atom($1);"],
            ["( expression )", "$$ = $2;"]
        ]
    }
});

if (typeof module === "object") module.exports = parsexp;
