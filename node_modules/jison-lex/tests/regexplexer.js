var RegExpLexer = require("../regexp-lexer"),
    assert = require("assert");

exports["test basic matchers"] = function() {
    var dict = {
        rules: [
           ["x", "return 'X';" ],
           ["y", "return 'Y';" ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = "xxyx";

    var lexer = new RegExpLexer(dict, input);
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "EOF");
};

exports["test set input after"] = function() {
    var dict = {
        rules: [
           ["x", "return 'X';" ],
           ["y", "return 'Y';" ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = "xxyx";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "EOF");
};

exports["test unrecognized char"] = function() {
    var dict = {
        rules: [
           ["x", "return 'X';" ],
           ["y", "return 'Y';" ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = "xa";

    var lexer = new RegExpLexer(dict, input);
    assert.equal(lexer.lex(), "X");
    assert.throws(function(){lexer.lex()}, "bad char");
};

exports["test macro"] = function() {
    var dict = {
        macros: {
            "digit": "[0-9]"
        },
        rules: [
           ["x", "return 'X';" ],
           ["y", "return 'Y';" ],
           ["{digit}+", "return 'NAT';" ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = "x12234y42";

    var lexer = new RegExpLexer(dict, input);
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "NAT");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "NAT");
    assert.equal(lexer.lex(), "EOF");
};

exports["test macro precedence"] = function() {
    var dict = {
        macros: {
            "hex": "[0-9]|[a-f]"
        },
        rules: [
           ["-", "return '-';" ],
           ["{hex}+", "return 'HEX';" ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = "129-abfe-42dc-ea12";

    var lexer = new RegExpLexer(dict, input);
    assert.equal(lexer.lex(), "HEX");
    assert.equal(lexer.lex(), "-");
    assert.equal(lexer.lex(), "HEX");
    assert.equal(lexer.lex(), "-");
    assert.equal(lexer.lex(), "HEX");
    assert.equal(lexer.lex(), "-");
    assert.equal(lexer.lex(), "HEX");
    assert.equal(lexer.lex(), "EOF");
};

exports["test nested macros"] = function () {
    var dict = {
        macros: {
            "digit": "[0-9]",
            "2digit": "{digit}{digit}",
            "3digit": "{2digit}{digit}"
        },
        rules: [
           ["x", "return 'X';" ],
           ["y", "return 'Y';" ],
           ["{3digit}", "return 'NNN';" ],
           ["{2digit}", "return 'NN';" ],
           ["{digit}", "return 'N';" ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = "x1y42y123";

    var lexer = new RegExpLexer(dict, input);
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "N");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "NN");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "NNN");
    assert.equal(lexer.lex(), "EOF");
};

exports["test nested macro precedence"] = function() {
    var dict = {
        macros: {
            "hex": "[0-9]|[a-f]",
            "col": "#{hex}+"
        },
        rules: [
           ["-", "return '-';" ],
           ["{col}", "return 'HEX';" ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = "#129-#abfe-#42dc-#ea12";

    var lexer = new RegExpLexer(dict, input);
    assert.equal(lexer.lex(), "HEX");
    assert.equal(lexer.lex(), "-");
    assert.equal(lexer.lex(), "HEX");
    assert.equal(lexer.lex(), "-");
    assert.equal(lexer.lex(), "HEX");
    assert.equal(lexer.lex(), "-");
    assert.equal(lexer.lex(), "HEX");
    assert.equal(lexer.lex(), "EOF");
};

exports["test action include"] = function() {
    var dict = {
        rules: [
           ["x", "return included ? 'Y' : 'N';" ],
           ["$", "return 'EOF';" ]
       ],
       actionInclude: "var included = true;"
    };

    var input = "x";

    var lexer = new RegExpLexer(dict, input);
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "EOF");
};

exports["test ignored"] = function() {
    var dict = {
        rules: [
           ["x", "return 'X';" ],
           ["y", "return 'Y';" ],
           ["\\s+", "/* skip whitespace */" ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = "x x   y x";

    var lexer = new RegExpLexer(dict, input);
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "EOF");
};

exports["test disambiguate"] = function() {
    var dict = {
        rules: [
           ["for\\b", "return 'FOR';" ],
           ["if\\b", "return 'IF';" ],
           ["[a-z]+", "return 'IDENTIFIER';" ],
           ["\\s+", "/* skip whitespace */" ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = "if forever for for";

    var lexer = new RegExpLexer(dict, input);
    assert.equal(lexer.lex(), "IF");
    assert.equal(lexer.lex(), "IDENTIFIER");
    assert.equal(lexer.lex(), "FOR");
    assert.equal(lexer.lex(), "FOR");
    assert.equal(lexer.lex(), "EOF");
};

exports["test yytext overwrite"] = function() {
    var dict = {
        rules: [
           ["x", "yytext = 'hi der'; return 'X';" ]
       ]
    };

    var input = "x";

    var lexer = new RegExpLexer(dict, input);
    lexer.lex();
    assert.equal(lexer.yytext, "hi der");
};

exports["test yylineno"] = function() {
    var dict = {
        rules: [
           ["\\s+", "/* skip whitespace */" ],
           ["x", "return 'x';" ],
           ["y", "return 'y';" ]
       ]
    };

    var input = "x\nxy\n\n\nx";

    var lexer = new RegExpLexer(dict, input);
    assert.equal(lexer.yylineno, 0);
    assert.equal(lexer.lex(), "x");
    assert.equal(lexer.lex(), "x");
    assert.equal(lexer.yylineno, 1);
    assert.equal(lexer.lex(), "y");
    assert.equal(lexer.yylineno, 1);
    assert.equal(lexer.lex(), "x");
    assert.equal(lexer.yylineno, 4);
};

exports["test yylloc"] = function() {
    var dict = {
        rules: [
           ["\\s+", "/* skip whitespace */" ],
           ["x", "return 'x';" ],
           ["y", "return 'y';" ]
       ]
    };

    var input = "x\nxy\n\n\nx";

    var lexer = new RegExpLexer(dict, input);
    assert.equal(lexer.lex(), "x");
    assert.equal(lexer.yylloc.first_column, 0);
    assert.equal(lexer.yylloc.last_column, 1);
    assert.equal(lexer.lex(), "x");
    assert.equal(lexer.yylloc.first_line, 2);
    assert.equal(lexer.yylloc.last_line, 2);
    assert.equal(lexer.yylloc.first_column, 0);
    assert.equal(lexer.yylloc.last_column, 1);
    assert.equal(lexer.lex(), "y");
    assert.equal(lexer.yylloc.first_line, 2);
    assert.equal(lexer.yylloc.last_line, 2);
    assert.equal(lexer.yylloc.first_column, 1);
    assert.equal(lexer.yylloc.last_column, 2);
    assert.equal(lexer.lex(), "x");
    assert.equal(lexer.yylloc.first_line, 5);
    assert.equal(lexer.yylloc.last_line, 5);
    assert.equal(lexer.yylloc.first_column, 0);
    assert.equal(lexer.yylloc.last_column, 1);
};

exports["test more()"] = function() {
    var dict = {
        rules: [
           ["x", "return 'X';" ],
           ['"[^"]*', function(){
               if(yytext.charAt(yyleng-1) == '\\') {
                   this.more();
               } else {
                   yytext += this.input(); // swallow end quote
                   return "STRING";
               }
            } ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = 'x"fgjdrtj\\"sdfsdf"x';

    var lexer = new RegExpLexer(dict, input);
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "STRING");
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "EOF");
};

exports["test defined token returns"] = function() {
    var tokens = {"2":"X", "3":"Y", "4":"EOF"};
    var dict = {
        rules: [
           ["x", "return 'X';" ],
           ["y", "return 'Y';" ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = "xxyx";

    var lexer = new RegExpLexer(dict, input, tokens);

    assert.equal(lexer.lex(), 2);
    assert.equal(lexer.lex(), 2);
    assert.equal(lexer.lex(), 3);
    assert.equal(lexer.lex(), 2);
    assert.equal(lexer.lex(), 4);
};

exports["test module generator"] = function() {
    var dict = {
        rules: [
           ["x", "return 'X';" ],
           ["y", "return 'Y';" ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = "xxyx";

    var lexer_ = new RegExpLexer(dict);
    var lexerSource = lexer_.generateModule();
    eval(lexerSource);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "EOF");
};

exports["test generator with more complex lexer"] = function() {
    var dict = {
        rules: [
           ["x", "return 'X';" ],
           ['"[^"]*', function(){
               if(yytext.charAt(yyleng-1) == '\\') {
                   this.more();
               } else {
                   yytext += this.input(); // swallow end quote
                   return "STRING";
               }
            } ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = 'x"fgjdrtj\\"sdfsdf"x';

    var lexer_ = new RegExpLexer(dict);
    var lexerSource = lexer_.generateModule();
    eval(lexerSource);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "STRING");
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "EOF");
};

exports["test commonjs module generator"] = function() {
    var dict = {
        rules: [
           ["x", "return 'X';" ],
           ["y", "return 'Y';" ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = "xxyx";

    var lexer_ = new RegExpLexer(dict);
    var lexerSource = lexer_.generateCommonJSModule();
    var exports = {};
    eval(lexerSource);
    exports.lexer.setInput(input);

    assert.equal(exports.lex(), "X");
    assert.equal(exports.lex(), "X");
    assert.equal(exports.lex(), "Y");
    assert.equal(exports.lex(), "X");
    assert.equal(exports.lex(), "EOF");
};

exports["test amd module generator"] = function() {
    var dict = {
        rules: [
           ["x", "return 'X';" ],
           ["y", "return 'Y';" ],
           ["$", "return 'EOF';" ]
       ]
    };

    var input = "xxyx";

    var lexer_ = new RegExpLexer(dict);
    var lexerSource = lexer_.generateCommonJSModule();

    var lexer;
    var define = function (_, fn) {
      lexer = fn();
    };

    eval(lexerSource);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "EOF");
};

exports["test DJ lexer"] = function() {
    var dict = {
    "lex": {
        "macros": {
            "digit": "[0-9]",
            "id": "[a-zA-Z][a-zA-Z0-9]*"
        },

        "rules": [
            ["//.*",       "/* ignore comment */"],
            ["main\\b",     "return 'MAIN';"],
            ["class\\b",    "return 'CLASS';"],
            ["extends\\b",  "return 'EXTENDS';"],
            ["nat\\b",      "return 'NATTYPE';"],
            ["if\\b",       "return 'IF';"],
            ["else\\b",     "return 'ELSE';"],
            ["for\\b",      "return 'FOR';"],
            ["printNat\\b", "return 'PRINTNAT';"],
            ["readNat\\b",  "return 'READNAT';"],
            ["this\\b",     "return 'THIS';"],
            ["new\\b",      "return 'NEW';"],
            ["var\\b",      "return 'VAR';"],
            ["null\\b",     "return 'NUL';"],
            ["{digit}+",   "return 'NATLITERAL';"],
            ["{id}",       "return 'ID';"],
            ["==",         "return 'EQUALITY';"],
            ["=",          "return 'ASSIGN';"],
            ["\\+",        "return 'PLUS';"],
            ["-",          "return 'MINUS';"],
            ["\\*",        "return 'TIMES';"],
            [">",          "return 'GREATER';"],
            ["\\|\\|",     "return 'OR';"],
            ["!",          "return 'NOT';"],
            ["\\.",        "return 'DOT';"],
            ["\\{",        "return 'LBRACE';"],
            ["\\}",        "return 'RBRACE';"],
            ["\\(",        "return 'LPAREN';"],
            ["\\)",        "return 'RPAREN';"],
            [";",          "return 'SEMICOLON';"],
            ["\\s+",       "/* skip whitespace */"],
            [".",          "print('Illegal character');throw 'Illegal character';"],
            ["$",          "return 'ENDOFFILE';"]
        ]
    }
};

    var input = "class Node extends Object { \
                      var nat value    var nat value;\
                      var Node next;\
                      var nat index;\
                    }\
\
                    class List extends Object {\
                      var Node start;\
\
                      Node prepend(Node startNode) {\
                        startNode.next = start;\
                        start = startNode;\
                      }\
\
                      nat find(nat index) {\
                        var nat value;\
                        var Node node;\
\
                        for(node = start;!(node == null);node = node.next){\
                          if(node.index == index){\
                            value = node.value;\
                          } else { 0; };\
                        };\
\
                        value;\
                      }\
                    }\
\
                    main {\
                      var nat index;\
                      var nat value;\
                      var List list;\
                      var Node startNode;\
\
                      index = readNat();\
                      list = new List;\
\
                      for(0;!(index==0);0){\
                        value = readNat();\
                        startNode = new Node;\
                        startNode.index = index;\
                        startNode.value = value;\
                        list.prepend(startNode);\
                        index = readNat();\
                      };\
\
                      index = readNat();\
\
                      for(0;!(index==0);0){\
                        printNat(list.find(index));\
                        index = readNat();\
                      };\
                    }";

    var lexer = new RegExpLexer(dict.lex);
    lexer.setInput(input);
    var tok;
    while (tok = lexer.lex(), tok!==1) {
        assert.equal(typeof tok, "string");
    }
};

exports["test instantiation from string"] = function() {
    var dict = "%%\n'x' {return 'X';}\n'y' {return 'Y';}\n<<EOF>> {return 'EOF';}";

    var input = "x";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "EOF");
};

exports["test inclusive start conditions"] = function() {
    var dict = {
        startConditions: {
            "TEST": 0,
        },
        rules: [
            ["enter-test", "this.begin('TEST');" ],
            [["TEST"], "x", "return 'T';" ],
            [["TEST"], "y", "this.begin('INITIAL'); return 'TY';" ],
            ["x", "return 'X';" ],
            ["y", "return 'Y';" ],
            ["$", "return 'EOF';" ]
        ]
    };
    var input = "xenter-testxyy";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "T");
    assert.equal(lexer.lex(), "TY");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "EOF");
};

exports["test exclusive start conditions"] = function() {
    var dict = {
        startConditions: {
            "EAT": 1,
        },
        rules: [
            ["//", "this.begin('EAT');" ],
            [["EAT"], ".", "" ],
            [["EAT"], "\\n", "this.begin('INITIAL');" ],
            ["x", "return 'X';" ],
            ["y", "return 'Y';" ],
            ["$", "return 'EOF';" ]
        ]
    };
    var input = "xy//yxteadh//ste\ny";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "EOF");
};

exports["test pop start condition stack"] = function() {
    var dict = {
        startConditions: {
            "EAT": 1,
        },
        rules: [
            ["//", "this.begin('EAT');" ],
            [["EAT"], ".", "" ],
            [["EAT"], "\\n", "this.popState();" ],
            ["x", "return 'X';" ],
            ["y", "return 'Y';" ],
            ["$", "return 'EOF';" ]
        ]
    };
    var input = "xy//yxteadh//ste\ny";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "EOF");
};


exports["test star start condition"] = function() {
    var dict = {
        startConditions: {
            "EAT": 1,
        },
        rules: [
            ["//", "this.begin('EAT');" ],
            [["EAT"], ".", "" ],
            ["x", "return 'X';" ],
            ["y", "return 'Y';" ],
            [["*"],"$", "return 'EOF';" ]
        ]
    };
    var input = "xy//yxteadh//stey";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "EOF");
};

exports["test start condition constants"] = function() {
    var dict = {
        startConditions: {
            "EAT": 1,
        },
        rules: [
            ["//", "this.begin('EAT');" ],
            [["EAT"], ".", "if (YYSTATE==='EAT') return 'E';" ],
            ["x", "if (YY_START==='INITIAL') return 'X';" ],
            ["y", "return 'Y';" ],
            [["*"],"$", "return 'EOF';" ]
        ]
    };
    var input = "xy//y";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "Y");
    assert.equal(lexer.lex(), "E");
    assert.equal(lexer.lex(), "EOF");
};

exports["test unicode encoding"] = function() {
    var dict = {
        rules: [
            ["\\u2713", "return 'CHECK';" ],
            ["\\u03c0", "return 'PI';" ],
            ["y", "return 'Y';" ]
        ]
    };
    var input = "\u2713\u03c0y";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "CHECK");
    assert.equal(lexer.lex(), "PI");
    assert.equal(lexer.lex(), "Y");
};

exports["test unicode"] = function() {
    var dict = {
        rules: [
            ["π", "return 'PI';" ],
            ["y", "return 'Y';" ]
        ]
    };
    var input = "πy";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "PI");
    assert.equal(lexer.lex(), "Y");
};

exports["test longest match returns"] = function() {
    var dict = {
        rules: [
            [".", "return 'DOT';" ],
            ["cat", "return 'CAT';" ]
        ],
        options: {flex: true}
    };
    var input = "cat!";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "CAT");
    assert.equal(lexer.lex(), "DOT");
};

exports["test case insensitivity"] = function() {
    var dict = {
        rules: [
            ["cat", "return 'CAT';" ]
        ],
        options: {'case-insensitive': true}
    };
    var input = "Cat";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "CAT");
};

exports["test less"] = function() {
    var dict = {
        rules: [
            ["cat", "this.less(2); return 'CAT';" ],
            ["t", "return 'T';" ]
        ],
    };
    var input = "cat";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "CAT");
    assert.equal(lexer.lex(), "T");
};

exports["test EOF unput"] = function() {
    var dict = {
        startConditions: {
            "UN": 1,
        },
        rules: [
            ["U", "this.begin('UN');return 'U';" ],
            [["UN"],"$", "this.unput('X')" ],
            [["UN"],"X", "this.popState();return 'X';" ],
            ["$", "return 'EOF'" ]
        ]
    };
    var input = "U";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "U");
    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "EOF");
};

exports["test flex mode default rule"] = function() {
    var dict = {
        rules: [
            ["x", "return 'X';" ]
        ],
        options: {flex: true}
    };
    var input = "xyx";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "X");
    assert.equal(lexer.lex(), "X");
};

exports["test pipe precedence"] = function() {
    var dict = {
        rules: [
            ["x|y", "return 'X_Y';" ],
            [".",   "return 'N';"]
        ]
    };
    var input = "xny";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "X_Y");
    assert.equal(lexer.lex(), "N");
    assert.equal(lexer.lex(), "X_Y");
};

exports["test ranges"] = function() {
    var dict = {
        rules: [
            ["x+", "return 'X';" ],
            [".",   "return 'N';"]
        ],
        options: {ranges: true}
    };
    var input = "xxxyy";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "X");
    assert.deepEqual(lexer.yylloc.range, [0, 3]);
};

exports["test unput location"] = function() {
    var dict = {
        rules: [
            ["x+", "return 'X';" ],
            ["y\\n", "this.unput('\\n'); return 'Y';" ],
            ["\\ny", "this.unput('y'); return 'BR';" ],
            ["y", "return 'Y';" ],
            [".",   "return 'N';"]
        ],
        options: {ranges: true}
    };
    var input = "xxxy\ny";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);
    console.log(lexer.rules);

    assert.equal(lexer.next(), "X");
    assert.deepEqual(lexer.yylloc, {first_line: 1,
                                    first_column: 0,
                                    last_line: 1,
                                    last_column: 3,
                                    range: [0, 3]});
    assert.equal(lexer.next(), "Y");
    assert.deepEqual(lexer.yylloc, {first_line: 1,
                                    first_column: 3,
                                    last_line: 1,
                                    last_column: 4,
                                    range: [3, 4]});
    assert.equal(lexer.next(), "BR");
    assert.deepEqual(lexer.yylloc, {first_line: 1,
                                    first_column: 4,
                                    last_line: 2,
                                    last_column: 0,
                                    range: [4, 5]});
    assert.equal(lexer.next(), "Y");
    assert.deepEqual(lexer.yylloc, {first_line: 2,
                                    first_column: 0,
                                    last_line: 2,
                                    last_column: 1,
                                    range: [5, 6]});

};

exports["test unput location again"] = function() {
    var dict = {
        rules: [
            ["x+", "return 'X';" ],
            ["y\\ny\\n", "this.unput('\\n'); return 'YY';" ],
            ["\\ny", "this.unput('y'); return 'BR';" ],
            ["y", "return 'Y';" ],
            [".",   "return 'N';"]
        ],
        options: {ranges: true}
    };
    var input = "xxxy\ny\ny";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);
    console.log(lexer.rules);

    assert.equal(lexer.next(), "X");
    assert.deepEqual(lexer.yylloc, {first_line: 1,
                                    first_column: 0,
                                    last_line: 1,
                                    last_column: 3,
                                    range: [0, 3]});
    assert.equal(lexer.next(), "YY");
    assert.deepEqual(lexer.yylloc, {first_line: 1,
                                    first_column: 3,
                                    last_line: 2,
                                    last_column: 1,
                                    range: [3, 6]});
    assert.equal(lexer.next(), "BR");
    assert.deepEqual(lexer.yylloc, {first_line: 2,
                                    first_column: 1,
                                    last_line: 3,
                                    last_column: 0,
                                    range: [6, 7]});
    assert.equal(lexer.next(), "Y");
    assert.deepEqual(lexer.yylloc, {first_line: 3,
                                    first_column: 0,
                                    last_line: 3,
                                    last_column: 1,
                                    range: [7, 8]});

};

exports["test backtracking lexer reject() method"] = function() {
    var dict = {
        rules: [
            ["[A-Z]+([0-9]+)", "if (this.matches[1].length) this.reject(); else return 'ID';" ],
            ["[A-Z]+", "return 'WORD';" ],
            ["[0-9]+", "return 'NUM';" ]
        ],
        options: {backtrack_lexer: true}
    };
    var input = "A5";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.equal(lexer.lex(), "WORD");
    assert.equal(lexer.lex(), "NUM");
};

exports["test lexer reject() exception when not in backtracking mode"] = function() {
    var dict = {
        rules: [
            ["[A-Z]+([0-9]+)", "if (this.matches[1].length) this.reject(); else return 'ID';" ],
            ["[A-Z]+", "return 'WORD';" ],
            ["[0-9]+", "return 'NUM';" ]
        ],
        options: {backtrack_lexer: false}
    };
    var input = "A5";

    var lexer = new RegExpLexer(dict);
    lexer.setInput(input);

    assert.throws(function() {
      lexer.lex();
    },
    function(err) {
      return (err instanceof Error) && /You can only invoke reject/.test(err);
    });
};

