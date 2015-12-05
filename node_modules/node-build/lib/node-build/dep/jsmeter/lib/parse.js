// parse.js
// Parser for Simplified JavaScript written in Simplified JavaScript
// From Top Down Operator Precedence
// http://javascript.crockford.com/tdop/index.html
// Douglas Crockford
// 2008-07-07

var make_parse = function () {
    var scope;
    var symbol_table = {};
    var token;
    var tokens;
    var token_nr;
    var nextComments = [ ];

    var itself = function () {
        return this;
    };

    var original_scope = {
        define: function (n) {
            var t = this.def[n.value];
            /*if (typeof t === "object") {
                n.error(t.reserved ? "Already reserved." : "Already defined.");
            }*/
            this.def[n.value] = n;
            n.reserved = false;
            n.nud      = itself;
            n.led      = null;
            n.std      = null;
            n.lbp      = 0;
            n.scope    = scope;
            return n;
        },
        find: function (n) {
            var e = this, o;
            while (true) {
                o = e.def[n];
                if (o && o.nud) {
                    return o;
                }
                e = e.parent;
                if (!e) {
                    if (!symbol_table.hasOwnProperty(n)) {
                        var s = symbol(n);
                        s.nud = function() {
                            return this;
                        }
                    }
                    return symbol_table[n];
                }
            }
        },
        pop: function () {
            scope = this.parent;
        },
        reserve: function (n) {
            if (n.arity !== "name" || n.reserved) {
                return;
            }
            var t = this.def[n.value];
            if (t) {
                if (t.reserved) {
                    return;
                }
                if (t.arity === "name") {
                    //n.error("Already defined.");
                }
            }
            this.def[n.value] = n;
            n.reserved = true;
        }
    };

    var new_scope = function () {
        var s = scope;
        scope = Object.create(original_scope);
        scope.def = {};
        scope.parent = s;
        return scope;
    };

    var advance = function (id) {
        var a, o, t, v, cl, cli;
        if (id && token.id !== id) {
            token.error("Expected '" + id + "'.");
        }
        if (token_nr >= tokens.length) {
            token = symbol_table["(end)"];
            return;
        }
        t = tokens[token_nr];
        token_nr += 1;
        v = t.value;
        a = t.type;
        if (a === "name") {
            o = scope.find(v);
        } else if (a === "operator") {
            o = symbol_table[v];
            if (!o) {
                t.error("Unknown operator.");
            }
        } else if (a === "string" || a ===  "number" || a === "regexp" || a === "regexpops") {
            o = symbol_table["(literal)"];
            a = "literal";
        } else if (a === "comment") {
            o = symbol_table["(comment)"];
        } else {
            t.error("Unexpected token.");
        }
        token = Object.create(o);
        token.from  = t.from;
        token.to    = t.to;
        token.line  = t.line;
        token.value = v;
        token.arity = a;
        //window.status = JSON.stringify(token);
        
        if (token.arity === "comment") {
            cl = v.split(/\n/g);
            for (cli=0; cli<cl.length; cli++) {
                nextComments.push(cl[cli]);
            }
            advance();
        }
        
        return token;
    };

    var expression = function (rbp) {
        var left;
        var t = token;
        advance();
        left = t.nud();
        while (rbp < token.lbp) {
            t = token;
            advance();
            left = t.led(left);
        }  
        if (left) {
            left.comments = nextComments;
            nextComments = [ ];
        }
        return left;
    };

    var statement = function () {
        var n = token, v;

        if (n.std) {
            advance();
            scope.reserve(n);
            return n.std();
        }
        v = expression(0);
        /*if (!v.assignment && 
            v.id !== "(" && 
            v.id!== "++" && 
            v.id!== "--" && 
            v.value!=="use strict" &&
            v.id!=="typeof") {
                v.error("Bad expression statement.");
        }*/
        /*if (v.assignment && v.arity==="function") {
            advance();
        } else {
            advance(";");
        }*/
        if (token.id===";") {
            advance(";");
        }  
        if (v) {
            v.comments = nextComments;
            nextComments = [ ];
        }
        return v;
    };

    var statements = function () {
        var a = [], s;
        while (true) {
            if (token.id === "}" || token.id === "(end)") {
                break;
            }
            s = statement();
            if (s) {
                a.push(s);
            }
        }
        return a.length === 0 ? null : a.length === 1 ? a[0] : a;
    };

    var block = function () {
        var t = token;
        advance("{");
        return t.std();
    };

    var original_symbol = {
        nud: function () {
            //this.error("Undefined.");
        },
        led: function (left) {
            this.error("Missing operator.");
        }
    };

    var symbol = function (id, bp) {
        var s = symbol_table[id];
        bp = bp || 0;
        if (s) {
            if (bp >= s.lbp) {
                s.lbp = bp;
            }
        } else {
            s = Object.create(original_symbol);
            s.id = s.value = id;
            s.lbp = bp;
            symbol_table[id] = s;
        }
        return s;
    };

    var constant = function (s, v) {
        var x = symbol(s);
        x.nud = function () {
            scope.reserve(this);
            this.value = symbol_table[this.id].value;
            this.arity = "literal";
            return this;
        };
        x.value = v;
        return x;
    };

    var infix = function (id, bp, led) {
        var s = symbol(id, bp);
        s.led = led || function (left) {
            this.first = left;
            this.second = expression(bp);
            this.arity = "binary";
            return this;
        };
        return s;
    };

    var infixr = function (id, bp, led) {
        var s = symbol(id, bp);
        s.led = led || function (left) {
            this.first = left;
            this.second = expression(bp - 1);
            this.arity = "binary";
            return this;
        };
        return s;
    };

    var assignment = function (id) {
        return infixr(id, 10, function (left) {
            if (left.id !== "." && left.id !== "[" && left.arity !== "name") {
                left.error("Bad lvalue.");
            }
            this.first = left;
            this.second = expression(9);
            this.assignment = true;
            this.arity = "binary";
            if (token.id===",") {
                advance(",");
            }
            return this;
        });
    };

    var prefix = function (id, nud) {
        var s = symbol(id);
        s.nud = nud || function () {
            scope.reserve(this);
            this.first = expression(70);
            this.arity = "unary";
            return this;
        };
        return s;
    };

    var stmt = function (s, f) {
        var x = symbol(s);
        x.std = f;
        return x;
    };

    symbol("(end)");
    symbol("(name)");
    symbol(":");
    symbol(";");
    symbol(")");
    symbol("]");
    symbol("}");
    symbol(",");
    symbol("else");

    constant("true", true);
    constant("false", false);
    constant("null", null);
    constant("pi", 3.141592653589793);
    constant("Object", {});
    constant("Array", []);
    constant("Date", "Date");
    constant("Math", "Math");

    symbol("(literal)").nud = itself;
    symbol("(comment)");

    symbol("this").nud = function () {
        scope.reserve(this);
        this.arity = "this";
        return this;
    };

    assignment("=");
    assignment("+=");
    assignment("-=");
    assignment("*=");
    assignment("/=");
    assignment("%=");
    assignment("&=");
    assignment("|=");
    assignment("^=");
    assignment(">>=");
    assignment(">>>=");
    assignment("<<=");

    infix("?", 20, function (left) {
        this.first = left;
        this.second = expression(0);
        advance(":");
        this.third = expression(0);
        this.arity = "ternary";
        return this;
    });

    infixr("&", 20);
    infixr("|", 20);
    
    infixr("&&", 30);
    infixr("||", 30);
    
    infixr("in", 40);
    infixr("==", 40);
    infixr("!=", 40);
    infixr("===", 40);
    infixr("!==", 40);
    infixr("<", 40);
    infixr("<=", 40);
    infixr(">", 40);
    infixr(">=", 40);
    infixr(">>", 40);
    infixr(">>>", 40);
    infixr("<<", 40);

    infixr("instanceof", 45);
    infix("+", 50);
    infix("-", 50);

    infix("^", 60);
    infix("*", 60);
    infix("/", 60);
    infix("%", 60);

    infix("++", 65, function (left) {
            this.first = left;
            this.arity = "unary";
            return this;
        });
    infix("--", 65, function (left) {
            this.first = left;
            this.arity = "unary";
            return this;
        });

    infix(".", 80, function (left) {
        this.first = left;
        //if (token.arity !== "name") {
        //    token.error("Expected a property name.");
        //}
        token.arity = "literal";
        this.second = token;
        this.arity = "binary";
        advance();
        return this;
    });

    infix("[", 80, function (left) {
        this.first = left;
        this.second = expression(0);
        this.arity = "binary";
        advance("]");
        return this;
    });

    infix("(", 80, function (left) {
        var a = [];
        if (left && (left.id === "." || left.id === "[")) {
            this.arity = "ternary";
            this.first = left.first;
            this.second = left.second;
            this.third = a;
        } else {
            this.arity = "binary";
            this.first = left;
            this.second = a;
            /*if ((left.arity !== "unary" || left.id !== "function") &&
                    left.arity !== "name" && left.id !== "(" &&
                    left.id !== "&&" && left.id !== "||" && left.id !== "?" &&
                    left.id !== "function") {
                left.error("Expected a variable name.");
            }*/
        }
        if (token.id !== ")") {
            while (true)  {
                a.push(expression(0));
                if (token.id !== ",") {
                    break;
                }
                advance(",");
            }
        }
        advance(")");
        return this;
    });

    prefix("new");

    prefix("!");
    prefix("~");
    prefix("-");
    prefix("+");
    prefix("--");
    prefix("++");
    prefix("typeof", function() {
        var e = expression(0);
        this.first = e;
        return this;
    });

    prefix("(", function () {
        var e = expression(0);
        advance(")");
        return e;
    });

    prefix("function", function () {
        var a = [];
        new_scope();
        if (token.arity === "name") {
            scope.define(token);
            this.name = token.value;
            advance();
        }
        if (token.id !== "(") {
            scope.define(token);
            this.name = token.value;
            advance();
        }
        advance("(");
        if (token.id !== ")") {
            while (true) {
                if (token.arity !== "name") {
                    token.error("Expected a parameter name.");
                }
                scope.define(token);
                a.push(token);
                advance();
                if (token.id !== ",") {
                    break;
                }
                advance(",");
            }
        }
        this.first = a;
        advance(")");
        this.second = block();
        /*advance("{");
        this.second = statements();
        advance("}");*/
        this.arity = "function";
        this.assignment = true;
        scope.pop();
        return this;
    });

    prefix("[", function () {
        var a = [];
        if (token.id !== "]") {
            while (true) {
                a.push(expression(0));
                if (token.id !== ",") {
                    break;
                }
                advance(",");
            }
        }
        advance("]");
        this.first = a;
        this.arity = "unary";
        return this;
    });

    prefix("{", function () {
        var a = [], n, v;
        if (token.id !== "}") {
            while (true) {
                n = token;
                if (n.arity !== "name" && n.arity !== "literal") {
                    token.error("Bad property name.");
                }
                advance();
                advance(":");
                v = expression(0);
                v.key = n.value;
                a.push(v);
                if (token.id !== ",") {
                    break;
                }
                advance(",");
            }
        }
        advance("}");
        this.first = a;
        this.arity = "unary";
        return this;
    });

    stmt("<script", function() {
        while (token.value!==">") {
            advance();
        }
        advance(">");
    });

    stmt("</script", function() {
        while (token.value!==">") {
            advance();
        }
        advance(">");
    });

    stmt("{", function () {
        new_scope();
        var a = statements();
        advance("}");
        scope.pop();
        return a;
    });

    stmt("var", function () {
        var a = [], n, t;
        while (true) {
            n = token;
            if (n.arity !== "name") {
                n.error("Expected a new variable name.");
            }
            scope.define(n);
            advance();
            if (token.id === "=") {
                t = token;
                advance("=");
                t.first = n;
                t.second = expression(0);
                t.arity = "binary";
                a.push(t);
            }
            if (token.id === "in") {
                t = token;
                advance("in");
                t.first = n;
                t.second = expression(0);
                t.arity = "binary";
                a.push(t);
            }
            if (token.id !== ",") {
                break;
            }
            advance(",");
        }
        if (token.id === ";") {
            advance(";");
        }
        return a.length === 0 ? null : a.length === 1 ? a[0] : a;
    });
    
    stmt("try", function() {
        this.first = block();
        if (token.value === "catch") {
            this.second = statement();
        }
        if (token.value === "finally") {
            //this.third = statement();
        }
        this.arity = "statement";
        return this;
    });
    
    stmt("catch", function() {
        advance("(");
        if (token.id!==")") {
            this.first = expression(0);
        }
        advance(")");
        this.second = block();
        this.arity = "statement";
        return this;
    });
    
    stmt("finally", function() {
        this.first = block();
        this.arity = "statement";
        return this;
    });

    stmt("if", function () {
        advance("(");
        this.first = expression(0);
        advance(")");
        if (token.value==="{") {
            this.second = block();
        } else {
            this.second = statement();
        }
        if (token.id === "else") {
            scope.reserve(token);
            advance("else");
            if (token.id==="if") {
                this.third = statement();
            } else if (token.value==="{") {
                this.third = block();
            } else {
                this.third = statement();
            }
            //this.third = token.id === "if" ? statement() : block();
        } else {
            this.third = null;
        }
        this.arity = "statement";
        return this;
    });
    
    stmt("debugger", function() {
        if (token.id === ";") {
            advance(";");
        }
        this.arity = "statement";
        return this;
    });

    stmt("return", function () {
        this.first = null;
        this.second = null;
        if (token.id !== ";") {
            this.first = expression(0);
        }
        if (token.id === ";") {
            advance(";");
        }
        this.arity = "statement";
        return this;
    });
    
    stmt("throw", function() {
        this.first = expression(0);
        if (token.id === ";") {
            advance(";");
        }
        this.arity = "statement";
        return this;
    });
    
    stmt("delete", function() {
        this.first = expression(0);
        if (token.id === ";") {
            advance(";");
        }
        this.arity = "statement";
        return this;
    });

    stmt("break", function () {
        if (token.id === ";") {
            advance(";");
        }
        if (token.id !== "}" && token.id !== "case" && token.id !== "default" && token.id !== "return") {
            //token.error("Unreachable statement.");
        }
        this.arity = "statement";
        return this;
    });

    stmt("while", function () {
        advance("(");
        this.first = expression(0);
        advance(")");
        if (token.value==="{") {
            this.second = block();
        } else {
            this.second = statement();
        }
        this.arity = "statement";
        return this;
    });
    
    stmt("switch", function() {
        advance("(");
        this.first = expression(0);
        advance(")");
        this.second = block();
        this.arity = "statement";
        return this;
    });
    
    stmt("case", function() {
        this.first = expression(0);
        advance(":");
        this.arity = "statement";
        return this;
    });
    
    stmt("default", function() {
        advance(":");
        this.arity = "statement";
        return this;
    });
    
    stmt("for", function() {
        this.first = [ ];
        advance("(");
        if (token.value==="var") {
            this.first.push(statement());
        } else if (token.value!==";") {
            while (token.id!==";" && token.id!==")") {
                this.first.push(expression(0));
            }
            if (token.id===";") {
                advance(";");
            }
        } else {
            advance(";");
        }
        while (token.id!==")") {
            if (token.value!==";") {
                this.first.push(expression(0));
                if (token.id===";") {
                    advance(";");
                }
            } else {
                advance(";");
            }
        }
        advance(")");
        if (token.value==="{") {
            this.second = block();
        } else {
            this.second = statement();
        }
        this.arity = "statement";
        return this;
    });

    return function (source) {
        tokens = source.tokens('=<>!+-*&|/%^', '=<>&|+-/');
        token_nr = 0;
        new_scope();
        advance();
        var s = statements();
        advance("(end)");
        scope.pop();
        if (s.length) {
            s[s.length-1].comments = nextComments;
        } else {
            s.comments = nextComments;
        }
        return s;
    };
};
