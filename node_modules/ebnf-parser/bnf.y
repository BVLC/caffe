%start spec

/* grammar for parsing jison grammar files */

%{
var transform = require('./ebnf-transform').transform;
var ebnf = false;
%}

%%

spec
    : declaration_list '%%' grammar optional_end_block EOF
        {
          $$ = $1;
          return extend($$, $3);
        }
    | declaration_list '%%' grammar '%%' CODE EOF
        {
          $$ = $1;
          yy.addDeclaration($$, { include: $5 });
          return extend($$, $3);
        }
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
    | parse_param
        {$$ = {parseParam: $1};}
    | options
        {$$ = {options: $1};}
    ;

options
    : OPTIONS token_list
        {$$ = $2;}
    ;

parse_param
    : PARSE_PARAM token_list
        {$$ = $2;}
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
        {
            $$ = $1;
            if ($2[0] in $$) 
                $$[$2[0]] = $$[$2[0]].concat($2[1]);
            else
                $$[$2[0]] = $2[1];
        }
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
        {
            $$ = [($1.length ? $1.join(' ') : '')];
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
    : expression suffix ALIAS
        {$$ = $expression + $suffix + "[" + $ALIAS + "]"; }
    | expression suffix
        {$$ = $expression + $suffix; }
    ;

expression
    : ID
        {$$ = $1; }
    | STRING
        {$$ = ebnf ? "'" + $1 + "'" : $1; }
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
        {$$ = '$$ =' + $1 + ';';}
    |
        {$$ = '';}
    ;

action_body
    :
        {$$ = '';}
    | action_comments_body
        {$$ = $1;}
    | action_body '{' action_body '}' action_comments_body
        {$$ = $1 + $2 + $3 + $4 + $5;}
    | action_body '{' action_body '}'
        {$$ = $1 + $2 + $3 + $4;}
    ;

action_comments_body
    : ACTION_BODY
        { $$ = yytext; }
    | action_comments_body ACTION_BODY
        { $$ = $1+$2; }
    ;

%%

// transform ebnf to bnf if necessary
function extend (json, grammar) {
    json.bnf = ebnf ? transform(grammar) : grammar;
    return json;
}

