var Jison = require("../setup").Jison,
    Lexer = require("../setup").Lexer,
    assert = require("assert");

exports["test BNF parser"] = function () {
    var grammar = {
        "lex": {
            "rules": [
              ["\\s+", "/* skip whitespace */"],
              ["[a-zA-Z][a-zA-Z0-9_-]*", "return 'ID';"],
              ["\"[^\"]+\"", "yytext = yytext.substr(1, yyleng-2); return 'STRING';"],
              ["'[^']+'", "yytext = yytext.substr(1, yyleng-2); return 'STRING';"],
              [":", "return ':';"],
              [";", "return ';';"],
              ["\\|", "return '|';"],
              ["%%", "return '%%';"],
              ["%prec", "return 'PREC';"],
              ["%start", "return 'START';"],
              ["%left", "return 'LEFT';"],
              ["%right", "return 'RIGHT';"],
              ["%nonassoc", "return 'NONASSOC';"],
              ["\\{[^}]*\\}", "yytext = yytext.substr(1, yyleng-2); return 'ACTION';"],
              [".", "/* ignore bad characters */"],
              ["$", "return 'EOF';"]
            ]
        },
        "bnf": {
            "spec" :[[ "declaration_list %% grammar EOF", "$$ = $1; $$.bnf = $3; return $$;" ]],

            "declaration_list" :[[ "declaration_list declaration", "$$ = $1; yy.addDeclaration($$, $2);" ],
                                 [ "", "$$ = {};" ]],

            "declaration" :[[ "START id", "$$ = {start: $2};" ],
                            [ "operator", "$$ = {operator: $1};" ]],

            "operator" :[[ "associativity token_list", "$$ = [$1]; $$.push.apply($$, $2);" ]],

            "associativity" :[[ "LEFT", "$$ = 'left';" ],
                              [ "RIGHT", "$$ = 'right';" ],
                              [ "NONASSOC", "$$ = 'nonassoc';" ]],

            "token_list" :[[ "token_list symbol", "$$ = $1; $$.push($2);" ],
                           [ "symbol", "$$ = [$1];" ]],

            "grammar" :[[ "production_list", "$$ = $1;" ]],

            "production_list" :[[ "production_list production", "$$ = $1; $$[$2[0]] = $2[1];" ],
                                [ "production", "$$ = {}; $$[$1[0]] = $1[1];" ]],

            "production" :[[ "id : handle_list ;", "$$ = [$1, $3];" ]],

            "handle_list" :[[ "handle_list | handle_action", "$$ = $1; $$.push($3);" ],
                            [ "handle_action", "$$ = [$1];" ]],

            "handle_action" :[[ "handle action prec", "$$ = [($1.length ? $1.join(' ') : '')]; if($2) $$.push($2); if($3) $$.push($3); if ($$.length === 1) $$ = $$[0];" ]],

            "handle" :[[ "handle symbol", "$$ = $1; $$.push($2)" ],
                       [ "", "$$ = [];" ]],

            "prec" :[[ "PREC symbol", "$$ = {prec: $2};" ],
                     [ "", "$$ = null;" ]],

            "symbol" :[[ "id", "$$ = $1;" ],
                       [ "STRING", "$$ = yytext;" ]],

            "id" :[[ "ID", "$$ = yytext;" ]],

            "action" :[[ "ACTION", "$$ = yytext;" ],
                       [ "", "$$ = '';" ]]
        }

    };

    var parser = new Jison.Parser(grammar);
    parser.yy.addDeclaration = function (grammar, decl) {
        if (decl.start) {
            grammar.start = decl.start
        }
        if (decl.operator) {
            if (!grammar.operators) {
                grammar.operators = [];
            }
            grammar.operators.push(decl.operator);
        }

    };

    var result = parser.parse('%start foo %left "+" "-" %right "*" "/" %nonassoc "=" STUFF %left UMINUS %% foo : bar baz blitz { stuff } %prec GEMINI | bar %prec UMINUS | ;\nbar: { things };\nbaz: | foo ;');
    assert.ok(result, "parse bnf production");
};

