// Basic Lexer implemented using JavaScript regular expressions
// MIT Licensed

var RegExpLexer = (function () {
"use strict";

var lexParser = require('lex-parser');
var version = require('./package.json').version;

// expand macros and convert matchers to RegExp's
function prepareRules(rules, macros, actions, tokens, startConditions, caseless) {
    var m,i,k,action,conditions,
        newRules = [];

    if (macros) {
        macros = prepareMacros(macros);
    }

    function tokenNumberReplacement (str, token) {
        return "return "+(tokens[token] || "'"+token+"'");
    }

    actions.push('switch($avoiding_name_collisions) {');

    for (i=0;i < rules.length; i++) {
        if (Object.prototype.toString.apply(rules[i][0]) !== '[object Array]') {
            // implicit add to all inclusive start conditions
            for (k in startConditions) {
                if (startConditions[k].inclusive) {
                    startConditions[k].rules.push(i);
                }
            }
        } else if (rules[i][0][0] === '*') {
            // Add to ALL start conditions
            for (k in startConditions) {
                startConditions[k].rules.push(i);
            }
            rules[i].shift();
        } else {
            // Add to explicit start conditions
            conditions = rules[i].shift();
            for (k=0;k<conditions.length;k++) {
                startConditions[conditions[k]].rules.push(i);
            }
        }

        m = rules[i][0];
        if (typeof m === 'string') {
            for (k in macros) {
                if (macros.hasOwnProperty(k)) {
                    m = m.split("{"+k+"}").join('(' + macros[k] + ')');
                }
            }
            m = new RegExp("^(?:"+m+")", caseless ? 'i':'');
        }
        newRules.push(m);
        if (typeof rules[i][1] === 'function') {
            rules[i][1] = String(rules[i][1]).replace(/^\s*function \(\)\s?\{/, '').replace(/\}\s*$/, '');
        }
        action = rules[i][1];
        if (tokens && action.match(/return '[^']+'/)) {
            action = action.replace(/return '([^']+)'/g, tokenNumberReplacement);
        }
        actions.push('case '+i+':' +action+'\nbreak;');
    }
    actions.push("}");

    return newRules;
}

// expand macros within macros
function prepareMacros (macros) {
    var cont = true,
        m,i,k,mnew;
    while (cont) {
        cont = false;
        for (i in macros) if (macros.hasOwnProperty(i)) {
            m = macros[i];
            for (k in macros) if (macros.hasOwnProperty(k) && i !== k) {
                mnew = m.split("{"+k+"}").join('(' + macros[k] + ')');
                if (mnew !== m) {
                    cont = true;
                    macros[i] = mnew;
                }
            }
        }
    }
    return macros;
}

function prepareStartConditions (conditions) {
    var sc,
        hash = {};
    for (sc in conditions) if (conditions.hasOwnProperty(sc)) {
        hash[sc] = {rules:[],inclusive:!!!conditions[sc]};
    }
    return hash;
}

function buildActions (dict, tokens) {
    var actions = [dict.actionInclude || '', "var YYSTATE=YY_START;"];
    var tok;
    var toks = {};

    for (tok in tokens) {
        toks[tokens[tok]] = tok;
    }

    if (dict.options && dict.options.flex) {
        dict.rules.push([".", "console.log(yytext);"]);
    }

    this.rules = prepareRules(dict.rules, dict.macros, actions, tokens && toks, this.conditions, this.options["case-insensitive"]);
    var fun = actions.join("\n");
    "yytext yyleng yylineno yylloc".split(' ').forEach(function (yy) {
        fun = fun.replace(new RegExp("\\b("+yy+")\\b", "g"), "yy_.$1");
    });


    // first try to create the performAction function the old way,
    // but this will break for some legal constructs in the user action code:
    try {
        return Function("yy,yy_,$avoiding_name_collisions,YY_START", fun);
    } catch (e) {
        return "function anonymous(yy,yy_,$avoiding_name_collisions,YY_START) {" + fun + "\n}";
    }
}

function RegExpLexer (dict, input, tokens) {
    if (typeof dict === 'string') {
        dict = lexParser.parse(dict);
    }
    dict = dict || {};
    this.options = dict.options || {};

    this.conditions = prepareStartConditions(dict.startConditions);
    this.conditions.INITIAL = {rules:[],inclusive:true};

    this.performAction = buildActions.call(this, dict, tokens);
    this.conditionStack = ['INITIAL'];

    this.moduleInclude = (dict.moduleInclude || '').trim();

    this.yy = {};
    if (input) {
        this.setInput(input);
    }
}

RegExpLexer.prototype = {
    EOF: 1,
    parseError: function parseError(str, hash) {
        if (this.yy.parser) {
            this.yy.parser.parseError(str, hash);
        } else {
            throw new Error(str);
        }
    },

    // resets the lexer, sets new input
    setInput: function (input) {
        this._input = input;
        this._more = this._backtrack = this.done = false;
        this.yylineno = this.yyleng = 0;
        this.yytext = this.matched = this.match = '';
        this.conditionStack = ['INITIAL'];
        this.yylloc = {
            first_line: 1,
            first_column: 0,
            last_line: 1,
            last_column: 0
        };
        if (this.options.ranges) {
            this.yylloc.range = [0,0];
        }
        this.offset = 0;
        return this;
    },

    // consumes and returns one char from the input
    input: function () {
        var ch = this._input[0];
        this.yytext += ch;
        this.yyleng++;
        this.offset++;
        this.match += ch;
        this.matched += ch;
        var lines = ch.match(/(?:\r\n?|\n).*/g);
        if (lines) {
            this.yylineno++;
            this.yylloc.last_line++;
        } else {
            this.yylloc.last_column++;
        }
        if (this.options.ranges) {
            this.yylloc.range[1]++;
        }

        this._input = this._input.slice(1);
        return ch;
    },

    // unshifts one char (or a string) into the input
    unput: function (ch) {
        var len = ch.length;
        var lines = ch.split(/(?:\r\n?|\n)/g);

        this._input = ch + this._input;
        this.yytext = this.yytext.substr(0, this.yytext.length - len - 1);
        //this.yyleng -= len;
        this.offset -= len;
        var oldLines = this.match.split(/(?:\r\n?|\n)/g);
        this.match = this.match.substr(0, this.match.length - 1);
        this.matched = this.matched.substr(0, this.matched.length - 1);

        if (lines.length - 1) {
            this.yylineno -= lines.length - 1;
        }
        var r = this.yylloc.range;

        this.yylloc = {
            first_line: this.yylloc.first_line,
            last_line: this.yylineno + 1,
            first_column: this.yylloc.first_column,
            last_column: lines ?
                (lines.length === oldLines.length ? this.yylloc.first_column : 0)
                 + oldLines[oldLines.length - lines.length].length - lines[0].length :
              this.yylloc.first_column - len
        };

        if (this.options.ranges) {
            this.yylloc.range = [r[0], r[0] + this.yyleng - len];
        }
        this.yyleng = this.yytext.length;
        return this;
    },

    // When called from action, caches matched text and appends it on next action
    more: function () {
        this._more = true;
        return this;
    },

    // When called from action, signals the lexer that this rule fails to match the input, so the next matching rule (regex) should be tested instead.
    reject: function () {
        if (this.options.backtrack_lexer) {
            this._backtrack = true;
        } else {
            return this.parseError('Lexical error on line ' + (this.yylineno + 1) + '. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).\n' + this.showPosition(), {
                text: "",
                token: null,
                line: this.yylineno
            });

        }
        return this;
    },

    // retain first n characters of the match
    less: function (n) {
        this.unput(this.match.slice(n));
    },

    // displays already matched input, i.e. for error messages
    pastInput: function () {
        var past = this.matched.substr(0, this.matched.length - this.match.length);
        return (past.length > 20 ? '...':'') + past.substr(-20).replace(/\n/g, "");
    },

    // displays upcoming input, i.e. for error messages
    upcomingInput: function () {
        var next = this.match;
        if (next.length < 20) {
            next += this._input.substr(0, 20-next.length);
        }
        return (next.substr(0,20) + (next.length > 20 ? '...' : '')).replace(/\n/g, "");
    },

    // displays the character position where the lexing error occurred, i.e. for error messages
    showPosition: function () {
        var pre = this.pastInput();
        var c = new Array(pre.length + 1).join("-");
        return pre + this.upcomingInput() + "\n" + c + "^";
    },

    // test the lexed token: return FALSE when not a match, otherwise return token
    test_match: function(match, indexed_rule) {
        var token,
            lines,
            backup;

        if (this.options.backtrack_lexer) {
            // save context
            backup = {
                yylineno: this.yylineno,
                yylloc: {
                    first_line: this.yylloc.first_line,
                    last_line: this.last_line,
                    first_column: this.yylloc.first_column,
                    last_column: this.yylloc.last_column
                },
                yytext: this.yytext,
                match: this.match,
                matches: this.matches,
                matched: this.matched,
                yyleng: this.yyleng,
                offset: this.offset,
                _more: this._more,
                _input: this._input,
                yy: this.yy,
                conditionStack: this.conditionStack.slice(0),
                done: this.done
            };
            if (this.options.ranges) {
                backup.yylloc.range = this.yylloc.range.slice(0);
            }
        }

        lines = match[0].match(/(?:\r\n?|\n).*/g);
        if (lines) {
            this.yylineno += lines.length;
        }
        this.yylloc = {
            first_line: this.yylloc.last_line,
            last_line: this.yylineno + 1,
            first_column: this.yylloc.last_column,
            last_column: lines ?
                         lines[lines.length - 1].length - lines[lines.length - 1].match(/\r?\n?/)[0].length :
                         this.yylloc.last_column + match[0].length
        };
        this.yytext += match[0];
        this.match += match[0];
        this.matches = match;
        this.yyleng = this.yytext.length;
        if (this.options.ranges) {
            this.yylloc.range = [this.offset, this.offset += this.yyleng];
        }
        this._more = false;
        this._backtrack = false;
        this._input = this._input.slice(match[0].length);
        this.matched += match[0];
        token = this.performAction.call(this, this.yy, this, indexed_rule, this.conditionStack[this.conditionStack.length - 1]);
        if (this.done && this._input) {
            this.done = false;
        }
        if (token) {
            return token;
        } else if (this._backtrack) {
            // recover context
            for (var k in backup) {
                this[k] = backup[k];
            }
            return false; // rule action called reject() implying the next rule should be tested instead.
        }
        return false;
    },

    // return next match in input
    next: function () {
        if (this.done) {
            return this.EOF;
        }
        if (!this._input) {
            this.done = true;
        }

        var token,
            match,
            tempMatch,
            index;
        if (!this._more) {
            this.yytext = '';
            this.match = '';
        }
        var rules = this._currentRules();
        for (var i = 0; i < rules.length; i++) {
            tempMatch = this._input.match(this.rules[rules[i]]);
            if (tempMatch && (!match || tempMatch[0].length > match[0].length)) {
                match = tempMatch;
                index = i;
                if (this.options.backtrack_lexer) {
                    token = this.test_match(tempMatch, rules[i]);
                    if (token !== false) {
                        return token;
                    } else if (this._backtrack) {
                        match = false;
                        continue; // rule action called reject() implying a rule MISmatch.
                    } else {
                        // else: this is a lexer rule which consumes input without producing a token (e.g. whitespace)
                        return false;
                    }
                } else if (!this.options.flex) {
                    break;
                }
            }
        }
        if (match) {
            token = this.test_match(match, rules[index]);
            if (token !== false) {
                return token;
            }
            // else: this is a lexer rule which consumes input without producing a token (e.g. whitespace)
            return false;
        }
        if (this._input === "") {
            return this.EOF;
        } else {
            return this.parseError('Lexical error on line ' + (this.yylineno + 1) + '. Unrecognized text.\n' + this.showPosition(), {
                text: "",
                token: null,
                line: this.yylineno
            });
        }
    },

    // return next match that has a token
    lex: function lex () {
        var r = this.next();
        if (r) {
            return r;
        } else {
            return this.lex();
        }
    },

    // activates a new lexer condition state (pushes the new lexer condition state onto the condition stack)
    begin: function begin (condition) {
        this.conditionStack.push(condition);
    },

    // pop the previously active lexer condition state off the condition stack
    popState: function popState () {
        var n = this.conditionStack.length - 1;
        if (n > 0) {
            return this.conditionStack.pop();
        } else {
            return this.conditionStack[0];
        }
    },

    // produce the lexer rule set which is active for the currently active lexer condition state
    _currentRules: function _currentRules () {
        if (this.conditionStack.length && this.conditionStack[this.conditionStack.length - 1]) {
            return this.conditions[this.conditionStack[this.conditionStack.length - 1]].rules;
        } else {
            return this.conditions["INITIAL"].rules;
        }
    },

    // return the currently active lexer condition state; when an index argument is provided it produces the N-th previous condition state, if available
    topState: function topState (n) {
        n = this.conditionStack.length - 1 - Math.abs(n || 0);
        if (n >= 0) {
            return this.conditionStack[n];
        } else {
            return "INITIAL";
        }
    },

    // alias for begin(condition)
    pushState: function pushState (condition) {
        this.begin(condition);
    },

    // return the number of states pushed
    stateStackSize: function stateStackSize() {
        return this.conditionStack.length;
    },

    generate:  function generate(opt) {
        var code = "";
        if (opt.moduleType === 'commonjs') {
            code = this.generateCommonJSModule(opt);
        } else if (opt.moduleType === 'amd') {
            code = this.generateAMDModule(opt);
        } else {
            code = this.generateModule(opt);
        }

        return code;
    },
    generateModuleBody: function generateModule() {
        var function_descriptions = {
            setInput: "resets the lexer, sets new input",
            input: "consumes and returns one char from the input",
            unput: "unshifts one char (or a string) into the input",
            more: "When called from action, caches matched text and appends it on next action",
            reject: "When called from action, signals the lexer that this rule fails to match the input, so the next matching rule (regex) should be tested instead.",
            less: "retain first n characters of the match",
            pastInput: "displays already matched input, i.e. for error messages",
            upcomingInput: "displays upcoming input, i.e. for error messages",
            showPosition: "displays the character position where the lexing error occurred, i.e. for error messages",
            test_match: "test the lexed token: return FALSE when not a match, otherwise return token",
            next: "return next match in input",
            lex: "return next match that has a token",
            begin: "activates a new lexer condition state (pushes the new lexer condition state onto the condition stack)",
            popState: "pop the previously active lexer condition state off the condition stack",
            _currentRules: "produce the lexer rule set which is active for the currently active lexer condition state",
            topState: "return the currently active lexer condition state; when an index argument is provided it produces the N-th previous condition state, if available",
            pushState: "alias for begin(condition)",
            stateStackSize: "return the number of states currently on the stack"
        };
        var out = "{\n";
        var p = [];
        var descr;
        for (var k in RegExpLexer.prototype) {
            if (RegExpLexer.prototype.hasOwnProperty(k) && k.indexOf("generate") === -1) {
                // copy the function description as a comment before the implementation; supports multi-line descriptions
                descr = "\n";
                if (function_descriptions[k]) {
                    descr += "// " + function_descriptions[k].replace(/\n/g, "\n\/\/ ") + "\n";
                }
                p.push(descr + k + ":" + (RegExpLexer.prototype[k].toString() || '""'));
            }
        }
        out += p.join(",\n");

        if (this.options) {
            out += ",\noptions: " + JSON.stringify(this.options);
        }

        out += ",\nperformAction: " + String(this.performAction);
        out += ",\nrules: [" + this.rules + "]";
        out += ",\nconditions: " + JSON.stringify(this.conditions);
        out += "\n}";

        return out;
    },
    generateModule: function generateModule(opt) {
        opt = opt || {};

        var out = "/* generated by jison-lex " + version + " */";
        var moduleName = opt.moduleName || "lexer";

        out += "\nvar " + moduleName + " = (function(){\nvar lexer = "
              + this.generateModuleBody();

        if (this.moduleInclude) out += ";\n"+this.moduleInclude;
        out += ";\nreturn lexer;\n})();";
        return out;
    },
    generateAMDModule: function generateAMDModule() {
        var out = "/* generated by jison-lex " + version + " */";

        out += "define([], function(){\nvar lexer = "
              + this.generateModuleBody();

        if (this.moduleInclude) out += ";\n"+this.moduleInclude;
        out += ";\nreturn lexer;"
             + "\n})();";
        return out;
    },
    generateCommonJSModule: function generateCommonJSModule(opt) {
        opt = opt || {};

        var out = "";
        var moduleName = opt.moduleName || "lexer";

        out += this.generateModule(opt);
        out += "\nexports.lexer = "+moduleName;
        out += ";\nexports.lex = function () { return "+moduleName+".lex.apply(lexer, arguments); };";
        return out;
    }
};

return RegExpLexer;

})();

module.exports = RegExpLexer;

