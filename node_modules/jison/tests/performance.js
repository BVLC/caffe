// TODO: ...should probably have some real performance tests.

var Jison = require("./setup").Jison;

var grammar = {
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
    },

    "tokens": "MAIN CLASS EXTENDS NATTYPE IF ELSE FOR PRINTNAT READNAT THIS NEW VAR NUL NATLITERAL ID ASSIGN PLUS MINUS TIMES EQUALITY GREATER OR NOT DOT SEMICOLON LBRACE RBRACE LPAREN RPAREN ENDOFFILE",
    "operators": [
        ["right", "ASSIGN"],
        ["left", "OR"],
        ["nonassoc", "EQUALITY", "GREATER"],
        ["left", "PLUS", "MINUS"],
        ["left", "TIMES"],
        ["right", "NOT"],
        ["left", "DOT"]
    ],

    "bnf": {
        "pgm": ["cdl MAIN LBRACE vdl el RBRACE ENDOFFILE"],

        "cdl": ["c cdl",
                ""],

        "c": ["CLASS id EXTENDS id LBRACE vdl mdl RBRACE"],

        "vdl": ["VAR t id SEMICOLON vdl",
                ""],

        "mdl": ["t id LPAREN t id RPAREN LBRACE vdl el RBRACE mdl",
                ""],

        "t": ["NATTYPE",
              "id"],

        "id": ["ID"],

        "el": ["e SEMICOLON el",
               "e SEMICOLON"],

        "e": ["NATLITERAL",
              "NUL",
              "id",
              "NEW id",
              "THIS", 
              "IF LPAREN e RPAREN LBRACE el RBRACE ELSE LBRACE el RBRACE ",
              "FOR LPAREN e SEMICOLON e SEMICOLON e RPAREN LBRACE el RBRACE",
              "READNAT LPAREN RPAREN",
              "PRINTNAT LPAREN e RPAREN",
              "e PLUS e",
              "e MINUS e",
              "e TIMES e",
              "e EQUALITY e",
              "e GREATER e",
              "NOT e",
              "e OR e",
              "e DOT id",
              "id ASSIGN e",
              "e DOT id ASSIGN e",
              "id LPAREN e RPAREN",
              "e DOT id LPAREN e RPAREN",
              "LPAREN e RPAREN"]
    }
};

var parser = new Jison.Parser(grammar, {type: 'lalr'});

