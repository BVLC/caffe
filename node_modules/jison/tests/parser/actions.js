var Jison = require("../setup").Jison,
    RegExpLexer = require("../setup").RegExpLexer,
    assert = require("assert");

exports["test Semantic action basic return"] = function() {
    var lexData = {
        rules: [
           ["x", "return 'x';"],
           ["y", "return 'y';"]
        ]
    };
    var grammar = {
        bnf: {
            "E"   :[ ["E x", "return 0"],
                     ["E y", "return 1"],
                     "" ]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData);

    assert.equal(parser.parse('x'), 0, "semantic action");
    assert.equal(parser.parse('y'), 1, "semantic action");
};

exports["test return null"] = function() {
    var lexData = {
        rules: [
           ["x", "return 'x';"]
        ]
    };
    var grammar = {
        bnf: {
            "E"   :[ ["E x", "return null;"],
                     "" ]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData);

    assert.equal(parser.parse('x'), null, "semantic action");
};

exports["test terminal semantic values are not null"] = function() {
    var lexData = {
        rules: [
           ["x", "return 'x';"],
           ["y", "return 'y';"]
        ]
    };
    var grammar = {
        bnf: {
            "E"   :[ ["E x", "return [$2 === 'x']"],
                     ["E y", "return [$2]"],
                     "" ]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData);

    assert.deepEqual(parser.parse('x'), [true], "semantic action");
    assert.deepEqual(parser.parse('y'), ['y'], "semantic action");
};

exports["test Semantic action stack lookup"] = function() {
    var lexData = {
        rules: [
           ["x", "return 'x';"],
           ["y", "return 'y';"]
        ]
    };
    var grammar = {
        bnf: {
            "pgm" :[ ["E", "return $1"] ],
            "E"   :[ ["B E", "return $1+$2"],
                      ["x", "$$ = 'EX'"] ],
            "B"   :[ ["y", "$$ = 'BY'"] ]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData);

    assert.equal(parser.parse('x'), "EX", "return first token");
    assert.equal(parser.parse('yx'), "BYEX", "return first after reduction");
};

exports["test Semantic actions on nullable grammar"] = function() {
    var lexData = {
        rules: [
           ["x", "return 'x';"]
        ]
    };
    var grammar = {
        bnf: {
            "S" :[ ["A", "return $1"] ],
            "A" :[ ['x A', "$$ = $2+'x'" ],
                   ['', "$$ = '->'" ] ]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData);

    assert.equal(parser.parse('xx'), "->xx", "return first after reduction");
};

exports["test named semantic value"] = function() {
    var lexData = {
        rules: [
           ["x", "return 'x';"]
        ]
    };
    var grammar = {
        bnf: {
            "S" :[ ["A", "return $A"] ],
            "A" :[ ['x A', "$$ = $A+'x'" ],
                   ['', "$$ = '->'" ] ]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData);

    assert.equal(parser.parse('xx'), "->xx", "return first after reduction");
};

exports["test ambiguous named semantic value"] = function() {
    var lexData = {
        rules: [
           ["x", "return 'x';"],
           ["y", "return 'y';"]
        ]
    };
    var grammar = {
        operators: [["left", "y"]],
        bnf: {
            "S" :[ ["A", "return $A"] ],
            "A" :[ ['A y A', "$$ = $A2+'y'+$A1" ],
                   ['x', "$$ = 'x'" ] ]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData);

    assert.equal(parser.parse('xyx'), "xyx", "return first after reduction");
};

exports["test Build AST"] = function() {
    var lexData = {
        rules: [
           ["x", "return 'x';"]
        ]
    };
    var grammar = {
        bnf: {
            "S" :[ ['A', "return $1;" ] ],
            "A" :[ ['x A', "$2.push(['ID',{value:'x'}]); $$ = $2;"],
                   ['', "$$ = ['A',{}];"] ]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData);

    var expectedAST = ['A',{},
        ['ID',{value:'x'}],
        ['ID',{value:'x'}],
        ['ID',{value:'x'}]];

    var r = parser.parse("xxx");
    assert.deepEqual(r, expectedAST);
};

exports["test 0+0 grammar"] = function() {
    var lexData2 = {
        rules: [
           ["0", "return 'ZERO';"],
           ["\\+", "return 'PLUS';"],
           ["$", "return 'EOF';"]
        ]
    };
    var grammar = {
        bnf: {
            "S" :[ [ "E EOF",    "return $1" ]],
            "E" :[ [ "E PLUS T", "$$ = ['+',$1,$3]"  ],
                   [ "T",        "$$ = $1" ]  ],
            "T" :[ [ "ZERO",     "$$ = [0]" ] ]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData2);

    var expectedAST = ["+", ["+", [0], [0]], [0]];

    assert.deepEqual(parser.parse("0+0+0"), expectedAST);
};

exports["test implicit $$ = $1 action"] = function() {
    var lexData2 = {
        rules: [
           ["0", "return 'ZERO';"],
           ["\\+", "return 'PLUS';"],
           ["$", "return 'EOF';"]
        ]
    };
    var grammar = {
        bnf: {
            "S" :[ [ "E EOF",    "return $1" ]],
            "E" :[ [ "E PLUS T", "$$ = ['+',$1,$3]"  ],
                   "T" ],
            "T" :[ [ "ZERO",     "$$ = [0]" ] ]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData2);

    var expectedAST = ["+", ["+", [0], [0]], [0]];

    assert.deepEqual(parser.parse("0+0+0"), expectedAST);
};

exports["test yytext"] = function() {
    var lexData = {
        rules: [
           ["x", "return 'x';"]
        ]
    };
    var grammar = {
        bnf: {
            "pgm" :[ ["Xexpr", "return $1;"] ],
            "Xexpr"   :[ ["x", "$$ = yytext;"] ]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData);

    assert.equal(parser.parse('x'), "x", "return first token");
};

exports["test yyleng"] = function() {
    var lexData = {
        rules: [
           ["x", "return 'x';"]
        ]
    };
    var grammar = {
        bnf: {
            "pgm" :[ ["Xexpr", "return $1;"] ],
            "Xexpr"   :[ ["x", "$$ = yyleng;"] ]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData);

    assert.equal(parser.parse('x'), 1, "return first token");
};

exports["test yytext more"] = function() {
    var lexData = {
        rules: [
           ["x", "return 'x';"],
           ["y", "return 'y';"]
        ]
    };
    var grammar = {
        bnf: {
            "pgm" :[ ["expr expr", "return $1+$2;"] ],
            "expr"   :[ ["x", "$$ = yytext;"],
                         ["y", "$$ = yytext;"] ]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData);

    assert.equal(parser.parse('xy'), "xy", "return first token");
};

exports["test action include"] = function() {
    var lexData = {
        rules: [
           ["y", "return 'y';"]
        ]
    };
    var grammar = {
        bnf: {
            "E"   :[ ["E y", "return test();"],
                     "" ]
        },
        actionInclude: function () {
            function test(val) {
                return 1;
            }
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData);

    assert.equal(parser.parse('y'), 1, "semantic action");
};

exports["test next token not shifted if only one action"] = function () {
    var lexData = {
        rules: [
           ["\\(", "return '(';"],
           ["\\)", "return ')';"],
           ["y", "return yy.xed ? 'yfoo' : 'ybar';"]
        ]
    };
    var grammar = {
        bnf: {
            "prog" :[ 'e ybar' ],
            "esub" :[[ '(', "yy.xed = true;" ]],
            "e" :[[ 'esub yfoo )', "yy.xed = false;" ]]
        }
    };

    var parser = new Jison.Parser(grammar);
    parser.lexer = new RegExpLexer(lexData);
    assert.ok(parser.parse('(y)y'), "should parse correctly");
};

