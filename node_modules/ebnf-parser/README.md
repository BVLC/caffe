# ebnf-parser

A parser for BNF and EBNF grammars used by jison.

## install

    npm install ebnf-parser


## build

To build the parser yourself, clone the git repo then run:

    make

This will generate `parser.js`, which is required by `ebnf-parser.js`.

## usage

The parser translates a string grammar or JSON grammar into a JSON grammar that jison can use (ENBF is transformed into BNF).

    var ebnfParser = require('ebnf-parser');

    // parse a bnf or ebnf string grammar
    ebnfParser.parse("%start ... %");

    // transform an ebnf JSON gramamr
    ebnfParser.transform({"ebnf": ...});


## example grammar

The parser can parse its own BNF grammar, shown below:

    %start spec

    /* grammar for parsing jison grammar files */

    %{
    var transform = require('./ebnf-transform').transform;
    var ebnf = false;
    %}

    %%

    spec
        : declaration_list '%%' grammar optional_end_block EOF
            {$$ = $1; return extend($$, $3);}
        | declaration_list '%%' grammar '%%' CODE EOF
            {$$ = $1; yy.addDeclaration($$,{include:$5}); return extend($$, $3);}
        ;

    optional_end_block
        :
        | '%%'
        ;

    declaration_list
        : declaration_list declaration
            {$$ = $1; yy.addDeclaration($$, $2);}
        |
            {$$ = {};}
        ;

    declaration
        : START id
            {$$ = {start: $2};}
        | LEX_BLOCK
            {$$ = {lex: $1};}
        | operator
            {$$ = {operator: $1};}
        | ACTION
            {$$ = {include: $1};}
        ;

    operator
        : associativity token_list
            {$$ = [$1]; $$.push.apply($$, $2);}
        ;

    associativity
        : LEFT
            {$$ = 'left';}
        | RIGHT
            {$$ = 'right';}
        | NONASSOC
            {$$ = 'nonassoc';}
        ;

    token_list
        : token_list symbol
            {$$ = $1; $$.push($2);}
        | symbol
            {$$ = [$1];}
        ;

    grammar
        : production_list
            {$$ = $1;}
        ;

    production_list
        : production_list production
            {$$ = $1;
              if($2[0] in $$) $$[$2[0]] = $$[$2[0]].concat($2[1]);
              else  $$[$2[0]] = $2[1];}
        | production
            {$$ = {}; $$[$1[0]] = $1[1];}
        ;

    production
        : id ':' handle_list ';'
            {$$ = [$1, $3];}
        ;

    handle_list
        : handle_list '|' handle_action
            {$$ = $1; $$.push($3);}
        | handle_action
            {$$ = [$1];}
        ;

    handle_action
        : handle prec action
            {$$ = [($1.length ? $1.join(' ') : '')];
                if($3) $$.push($3);
                if($2) $$.push($2);
                if ($$.length === 1) $$ = $$[0];
            }
        ;

    handle
        : handle expression_suffix
            {$$ = $1; $$.push($2)}
        |
            {$$ = [];}
        ;

    handle_sublist
        : handle_sublist '|' handle
            {$$ = $1; $$.push($3.join(' '));}
        | handle
            {$$ = [$1.join(' ')];}
        ;

    expression_suffix
        : expression suffix
            {$$ = $expression + $suffix; }
        ;

    expression
        : ID
            {$$ = $1; }
        | STRING
            {$$ = ebnf ? "'"+$1+"'" : $1; }
        | '(' handle_sublist ')'
            {$$ = '(' + $handle_sublist.join(' | ') + ')'; }
        ;

    suffix
        : {$$ = ''}
        | '*'
        | '?'
        | '+'
        ;

    prec
        : PREC symbol
            {$$ = {prec: $2};}
        |
            {$$ = null;}
        ;

    symbol
        : id
            {$$ = $1;}
        | STRING
            {$$ = yytext;}
        ;

    id
        : ID
            {$$ = yytext;}
        ;

    action
        : '{' action_body '}'
            {$$ = $2;}
        | ACTION
            {$$ = $1;}
        | ARROW_ACTION
            {$$ = '$$ ='+$1+';';}
        |
            {$$ = '';}
        ;

    action_body
        :
            {$$ = '';}
        | ACTION_BODY
            {$$ = yytext;}
        | action_body '{' action_body '}' ACTION_BODY
            {$$ = $1+$2+$3+$4+$5;}
        | action_body '{' action_body '}'
            {$$ = $1+$2+$3+$4;}
        ;

    %%

    // transform ebnf to bnf if necessary
    function extend (json, grammar) {
        json.bnf = ebnf ? transform(grammar) : grammar;
        return json;
    }

## license

MIT
