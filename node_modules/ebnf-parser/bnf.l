id                [a-zA-Z][a-zA-Z0-9_-]*


%x action code
%s bnf ebnf

%%

<bnf,ebnf>"%%"          this.pushState('code');return '%%';

<ebnf>"("               return '(';
<ebnf>")"               return ')';
<ebnf>"*"               return '*';
<ebnf>"?"               return '?';
<ebnf>"+"               return '+';

\s+                     /* skip whitespace */
"//".*                  /* skip comment */
"/*"(.|\n|\r)*?"*/"     /* skip comment */
"["{id}"]"              yytext = yytext.substr(1, yyleng-2); return 'ALIAS';
{id}                    return 'ID';
'"'[^"]+'"'             yytext = yytext.substr(1, yyleng-2); return 'STRING';
"'"[^']+"'"             yytext = yytext.substr(1, yyleng-2); return 'STRING';
":"                     return ':';
";"                     return ';';
"|"                     return '|';
"%%"                    this.pushState(ebnf ? 'ebnf' : 'bnf'); return '%%';
"%ebnf"                 if (!yy.options) yy.options = {}; ebnf = yy.options.ebnf = true;
"%prec"                 return 'PREC';
"%start"                return 'START';
"%left"                 return 'LEFT';
"%right"                return 'RIGHT';
"%nonassoc"             return 'NONASSOC';
"%parse-param"          return 'PARSE_PARAM';
"%options"              return 'OPTIONS';
"%lex"[\w\W]*?"/lex"    return 'LEX_BLOCK';
"%"[a-zA-Z]+[^\r\n]*    /* ignore unrecognized decl */
"<"[a-zA-Z]*">"         /* ignore type */
"{{"[\w\W]*?"}}"        yytext = yytext.substr(2, yyleng-4); return 'ACTION';
"%{"(.|\r|\n)*?"%}"     yytext = yytext.substr(2, yytext.length-4); return 'ACTION';
"{"                     yy.depth = 0; this.pushState('action'); return '{';
"->".*                  yytext = yytext.substr(2, yyleng-2); return 'ARROW_ACTION';
.                       /* ignore bad characters */
<*><<EOF>>              return 'EOF';

<action>"/*"(.|\n|\r)*?"*/"           return 'ACTION_BODY';
<action>"//".*                        return 'ACTION_BODY';
<action>"/"[^ /]*?['"{}'][^ ]*?"/"    return 'ACTION_BODY'; // regexp with braces or quotes (and no spaces)
<action>\"("\\\\"|'\"'|[^"])*\"       return 'ACTION_BODY';
<action>"'"("\\\\"|"\'"|[^'])*"'"     return 'ACTION_BODY';
<action>[/"'][^{}/"']+                return 'ACTION_BODY';
<action>[^{}/"']+                     return 'ACTION_BODY';
<action>"{"                           yy.depth++; return '{';
<action>"}"                           if (yy.depth==0) this.begin(ebnf ? 'ebnf' : 'bnf'); else yy.depth--; return '}';

<code>(.|\n|\r)+         return 'CODE';

%%
