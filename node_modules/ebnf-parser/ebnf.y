/* EBNF grammar spec */

%lex

id                        [a-zA-Z][a-zA-Z0-9_-]*

%%
\s+             /* skip whitespace */
{id}           return 'symbol';
"["{id}"]"     yytext = yytext.substr(1, yyleng-2); return 'ALIAS';
"'"[^']*"'"    return 'symbol';
"."            return 'symbol';

bar            return 'bar';
"("            return '(';
")"            return ')';
"*"            return '*';
"?"            return '?';
"|"            return '|';
"+"            return '+';
<<EOF>>        return 'EOF';
/lex

%start production

%%

production
  : handle EOF
    { return $handle; }
  ;

handle_list
  : handle
    { $$ = [$handle]; }
  | handle_list '|' handle
    { $handle_list.push($handle); }
  ;

handle
  :
    { $$ = []; }
  | handle expression_suffix
    { $handle.push($expression_suffix); }
  ;

expression_suffix
  : expression suffix ALIAS
    { $$ = ['xalias', $suffix, $expression, $ALIAS]; }
  | expression suffix
    { if ($suffix) $$ = [$suffix, $expression]; else $$ = $expression; }
  ;

expression
  : symbol
    { $$ = ['symbol', $symbol]; }
  | '(' handle_list ')'
    { $$ = ['()', $handle_list]; }
  ;

suffix
  : 
  | '*'
  | '?'
  | '+'
  ;
