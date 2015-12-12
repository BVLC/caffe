var Jison = require("../setup").Jison,
    Lexer = require("../setup").Lexer,
    assert = require("assert");

exports["test xx nullable grammar"] = function () {
    var lexData = {
        rules: [
           ["x", "return 'x';"],
           ["y", "return 'y';"]
        ]
    };
    var grammar = {
        tokens: [ 'x' ],
        startSymbol: "A",
        bnf: {
            "A" :[ 'A x',
                   ''      ]
        }
    };

    var parser = new Jison.Parser(grammar, {type: "lr"});
    parser.lexer = new Lexer(lexData);

    assert.ok(parser.parse("xxx"), "parse");
    assert.ok(parser.parse("x"), "parse single x");
    assert.throws(function (){parser.parse("+");}, "throws parse error on invalid");
};

exports["test LR parse"] = function () {
    var lexData2 = {
        rules: [
           ["0", "return 'ZERO';"],
           ["\\+", "return 'PLUS';"]
        ]
    };
    var grammar = {
        tokens: [ "ZERO", "PLUS"],
        startSymbol: "E",
        bnf: {
            "E" :[ "E PLUS T",
                   "T"      ],
            "T" :[ "ZERO" ]
        }
    };
    var parser = new Jison.Parser(grammar, {type: "lr"});
    parser.lexer = new Lexer(lexData2);

    assert.ok(parser.parse("0+0+0"), "parse");
};

exports["test basic JSON grammar"] = function () {
    var grammar = {
        "lex": {
            "macros": {
                "digit": "[0-9]"
            },
            "rules": [
                ["\\s+", "/* skip whitespace */"],
                ["{digit}+(\\.{digit}+)?", "return 'NUMBER';"],
                ["\"[^\"]*", function(){
                    if(yytext.charAt(yyleng-1) == '\\') {
                        // remove escape
                        yytext = yytext.substr(0,yyleng-2);
                        this.more();
                    } else {
                        yytext = yytext.substr(1); // swallow start quote
                        this.input(); // swallow end quote
                        return "STRING";
                    }
                }],
                ["\\{", "return '{'"],
                ["\\}", "return '}'"],
                ["\\[", "return '['"],
                ["\\]", "return ']'"],
                [",", "return ','"],
                [":", "return ':'"],
                ["true\\b", "return 'TRUE'"],
                ["false\\b", "return 'FALSE'"],
                ["null\\b", "return 'NULL'"]
            ]
        },

        "tokens": "STRING NUMBER { } [ ] , : TRUE FALSE NULL",
        "bnf": {
            "JsonThing": [ "JsonObject",
                           "JsonArray" ],

            "JsonObject": [ "{ JsonPropertyList }" ],

            "JsonPropertyList": [ "JsonProperty",
                                  "JsonPropertyList , JsonProperty" ],

            "JsonProperty": [ "StringLiteral : JsonValue" ],

            "JsonArray": [ "[ JsonValueList ]" ],

            "JsonValueList": [ "JsonValue",
                               "JsonValueList , JsonValue" ],

            "JsonValue": [ "StringLiteral",
                           "NumericalLiteral",
                           "JsonObject",
                           "JsonArray",
                           "TRUE",
                           "FALSE",
                           "NULL" ],

            "StringLiteral": [ "STRING" ],

            "NumericalLiteral": [ "NUMBER" ]
        },
    };

    var source = '{"foo": "Bar", "hi": 42, "array": [1,2,3.004,4], "false": false, "true":true, "null": null, "obj": {"ha":"ho"}, "string": "string\\"sgfg" }';

    var parser = new Jison.Parser(grammar, {type: "lr"});
    assert.ok(parser.parse(source));
}

exports["test compilers test grammar"] = function () {
    var lexData = {
        rules: [
           ["x", "return 'x';"]
        ]
    };
    var grammar = {
        tokens: [ 'x' ],
        startSymbol: "S",
        bnf: {
            "S" :[ 'A' ],
            "A" :[ 'B A', '' ],
            "B" :[ '', 'x' ]
        }
    };

    var parser = new Jison.Parser(grammar, {type: "lr"});
    parser.lexer = new Lexer(lexData);

    assert.ok(parser.parse("xxx"), "parse");
};

exports["test compilers test grammar 2"] = function () {
    var grammar = "%% n : a b ; a : | a x ; b : | b x y ;";

    var parser = new Jison.Generator(grammar, {type: "lr"});

    assert.equal(parser.conflicts, 1, "only one conflict");
};


