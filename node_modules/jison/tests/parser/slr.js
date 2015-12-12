var Jison = require("../setup").Jison,
    Lexer = require("../setup").Lexer,
    assert = require("assert");

var lexData = {
    rules: [
       ["x", "return 'x';"],
       ["y", "return 'y';"]
    ]
};

exports["test left-recursive nullable grammar"] = function () {

    var grammar = {
        tokens: [ 'x' ],
        startSymbol: "A",
        bnf: {
            "A" :[ 'A x',
                   ''      ]
        }
    };

    var gen = new Jison.Generator(grammar, {type: "slr"});
    var parser = gen.createParser();
    parser.lexer = new Lexer(lexData);

    assert.ok(parser.parse('xxx'), "parse 3 x's");
    assert.ok(parser.parse("x"), "parse single x");
    assert.throws(function(){parser.parse("y")}, "throws parse error on invalid token");
    assert.ok(gen.conflicts == 0, "no conflicts");
};

exports["test right-recursive nullable grammar"] = function () {

    var grammar = {
        tokens: [ 'x' ],
        startSymbol: "A",
        bnf: {
            "A" :[ 'x A',
                   ''      ]
        }
    };

    var gen = new Jison.Generator(grammar, {type: "slr"});
    var parser = gen.createParser();
    parser.lexer = new Lexer(lexData);

    assert.ok(parser.parse('xxx'), "parse 3 x's");
    assert.ok(gen.table.length == 4, "table has 4 states");
    assert.ok(gen.conflicts == 0, "no conflicts");
    assert.equal(gen.nullable('A'), true, "A is nullable");
};
