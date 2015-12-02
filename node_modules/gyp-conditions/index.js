var ometajs_ = require("ometajs");

var AbstractGrammar = ometajs_.grammars.AbstractGrammar;

var BSJSParser = ometajs_.grammars.BSJSParser;

var BSJSIdentity = ometajs_.grammars.BSJSIdentity;

var BSJSTranslator = ometajs_.grammars.BSJSTranslator;

var Condition = function Condition(source, opts) {
    AbstractGrammar.call(this, source, opts);
};

Condition.grammarName = "Condition";

Condition.match = AbstractGrammar.match;

Condition.matchAll = AbstractGrammar.matchAll;

exports.Condition = Condition;

require("util").inherits(Condition, AbstractGrammar);

Condition.prototype["spaces"] = function $spaces() {
    return this._seq(/^([ \t\v\f]+)/);
};

Condition.prototype["keyword"] = function $keyword() {
    return this._seq(/^(and|del|from|not|while|as|elif|global|or|with|assert|else|if|pass|yield|break|except|import|print|class|exec|in|raise|continue|finally|is|return|def|for|lambda|try)/);
};

Condition.prototype["literal"] = function $literal() {
    return this._atomic(function() {
        return this._rule("stringliteral", false, [], null, this["stringliteral"]);
    }) || this._atomic(function() {
        return this._rule("floatnumber", false, [], null, this["floatnumber"]);
    }) || this._atomic(function() {
        return this._rule("longinteger", false, [], null, this["longinteger"]);
    }) || this._atomic(function() {
        return this._rule("integer", false, [], null, this["integer"]);
    });
};

Condition.prototype["stringliteral"] = function $stringliteral() {
    var sp;
    return this._optional(function() {
        return this._rule("stringprefix", false, [], null, this["stringprefix"]);
    }) && (sp = this._getIntermediate(), true) && this._exec(this.stringMode = sp ? sp.toLowerCase() : "") && (this._atomic(function() {
        return this._rule("longstring", false, [], null, this["longstring"]);
    }) || this._atomic(function() {
        return this._rule("shortstring", false, [], null, this["shortstring"]);
    }));
};

Condition.prototype["stringprefix"] = function $stringprefix() {
    return this._seq("UR") || this._seq("Ur") || this._seq("uR") || this._seq("ur") || this._seq("br") || this._seq("Br") || this._seq("bR") || this._seq("BR") || this._match("r") || this._match("R") || this._match("u") || this._match("U") || this._match("b") || this._match("B");
};

Condition.prototype["shortstring"] = function $shortstring() {
    return this._atomic(function() {
        var v;
        return this._match("'") && this._any(function() {
            return this._atomic(function() {
                return this._rule("shortstringitem", false, [ "'" ], null, this["shortstringitem"]);
            });
        }) && (v = this._getIntermediate(), true) && this._match("'") && this._exec(v.join(""));
    }) || this._atomic(function() {
        var v;
        return this._match('"') && this._any(function() {
            return this._atomic(function() {
                return this._rule("shortstringitem", false, [ '"' ], null, this["shortstringitem"]);
            });
        }) && (v = this._getIntermediate(), true) && this._match('"') && this._exec(v.join(""));
    });
};

Condition.prototype["longstring"] = function $longstring() {
    return this._atomic(function() {
        var v;
        return this._rule("seq", false, [ "'''" ], null, this["seq"]) && this._any(function() {
            return this._atomic(function() {
                return !this._atomic(function() {
                    return this._rule("seq", false, [ "'''" ], null, this["seq"]);
                }, true) && this._rule("longstringitem", false, [], null, this["longstringitem"]);
            });
        }) && (v = this._getIntermediate(), true) && this._rule("seq", false, [ "'''" ], null, this["seq"]) && this._exec(v.join(""));
    }) || this._atomic(function() {
        var v;
        return this._rule("seq", false, [ '"""' ], null, this["seq"]) && this._any(function() {
            return this._atomic(function() {
                return !this._atomic(function() {
                    return this._rule("seq", false, [ '"""' ], null, this["seq"]);
                }, true) && this._rule("longstringitem", false, [], null, this["longstringitem"]);
            });
        }) && (v = this._getIntermediate(), true) && this._rule("seq", false, [ '"""' ], null, this["seq"]) && this._exec(v.join(""));
    });
};

Condition.prototype["shortstringitem"] = function $shortstringitem() {
    var q;
    return this._skip() && (q = this._getIntermediate(), true) && (this._atomic(function() {
        return this._rule("escapeseq", false, [], null, this["escapeseq"]);
    }) || this._atomic(function() {
        return this._rule("shortstringchar", false, [ q ], null, this["shortstringchar"]);
    }));
};

Condition.prototype["shortstringchar"] = function $shortstringchar() {
    var q;
    return this._skip() && (q = this._getIntermediate(), true) && !this._atomic(function() {
        return this._match("\\") || this._match("\n") || this._atomic(function() {
            return this._rule("seq", false, [ q ], null, this["seq"]);
        });
    }, true) && this._rule("char", false, [], null, this["char"]);
};

Condition.prototype["longstringitem"] = function $longstringitem() {
    return this._atomic(function() {
        return this._rule("escapeseq", false, [], null, this["escapeseq"]);
    }) || this._atomic(function() {
        return this._rule("longstringchar", false, [], null, this["longstringchar"]);
    });
};

Condition.prototype["longstringchar"] = function $longstringchar() {
    return !this._atomic(function() {
        return this._match("\\");
    }, true) && this._rule("char", false, [], null, this["char"]);
};

Condition.prototype["escapeseq"] = function $escapeseq() {
    return this._atomic(function() {
        var u;
        return this.stringMode.match(/u/) && this._rule("seq", false, [ "\\u" ], null, this["seq"]) && this._list(function() {
            return this._rule("digit", false, [], null, this["digit"]) && this._optional(function() {
                return this._rule("digit", false, [], null, this["digit"]);
            }) && this._optional(function() {
                return this._rule("digit", false, [], null, this["digit"]);
            }) && this._optional(function() {
                return this._rule("digit", false, [], null, this["digit"]);
            });
        }, true) && (u = this._getIntermediate(), true) && this._exec(String.fromCharCode(parseInt(u, 10)));
    }) || this._atomic(function() {
        var o;
        return !this.stringMode.match(/r/) && this._rule("seq", false, [ "\\" ], null, this["seq"]) && this._list(function() {
            return this._rule("octdigit", false, [], null, this["octdigit"]) && this._optional(function() {
                return this._rule("octdigit", false, [], null, this["octdigit"]);
            }) && this._optional(function() {
                return this._rule("octdigit", false, [], null, this["octdigit"]);
            });
        }, true) && (o = this._getIntermediate(), true) && this._exec(String.fromCharCode(parseInt(o, 8)));
    }) || this._atomic(function() {
        var x;
        return !this.stringMode.match(/r/) && this._rule("seq", false, [ "\\x" ], null, this["seq"]) && this._list(function() {
            return this._rule("hexdigit", false, [], null, this["hexdigit"]) && this._rule("hexdigit", false, [], null, this["hexdigit"]);
        }, true) && (x = this._getIntermediate(), true) && this._exec(String.fromCharCode(parseInt(x, 16)));
    }) || this._atomic(function() {
        var v;
        return !this.stringMode.match(/r/) && this._rule("seq", false, [ "\\" ], null, this["seq"]) && this._rule("char", false, [], null, this["char"]) && (v = this._getIntermediate(), 
        true) && this._exec(unescape(v));
    }) || this._atomic(function() {
        var v;
        return this.stringMode.match(/r/) && this._rule("seq", false, [ "\\" ], null, this["seq"]) && this._rule("char", false, [], null, this["char"]) && (v = this._getIntermediate(), 
        true) && this._exec("\\" + v);
    });
};

Condition.prototype["longinteger"] = function $longinteger() {
    var i;
    return this._rule("integer", false, [], null, this["integer"]) && (i = this._getIntermediate(), 
    true) && (this._match("L") || this._match("l")) && this._exec(i);
};

Condition.prototype["integer"] = function $integer() {
    return this._atomic(function() {
        return this._rule("bininteger", false, [], null, this["bininteger"]);
    }) || this._atomic(function() {
        return this._rule("octinteger", false, [], null, this["octinteger"]);
    }) || this._atomic(function() {
        return this._rule("hexinteger", false, [], null, this["hexinteger"]);
    }) || this._atomic(function() {
        return this._rule("decimalinteger", false, [], null, this["decimalinteger"]);
    });
};

Condition.prototype["decimalinteger"] = function $decimalinteger() {
    return this._atomic(function() {
        var v;
        return this._atomic(function() {
            return this._rule("nonzerodigit", false, [], null, this["nonzerodigit"]) && this._any(function() {
                return this._atomic(function() {
                    return this._rule("digit", false, [], null, this["digit"]);
                });
            });
        }) && (v = this._getIntermediate(), true) && this._exec(parseInt(v, 10));
    }) || this._atomic(function() {
        return this._match("0") && this._exec(0);
    });
};

Condition.prototype["octinteger"] = function $octinteger() {
    return this._atomic(function() {
        var v;
        return this._seq(/^(0o)/i) && this._many(function() {
            return this._atomic(function() {
                return this._rule("octdigit", false, [], null, this["octdigit"]);
            });
        }) && (v = this._getIntermediate(), true) && this._exec(parseInt(v.join(""), 8));
    }) || this._atomic(function() {
        var v;
        return this._atomic(function() {
            return this._match("0") && this._many(function() {
                return this._atomic(function() {
                    return this._rule("octdigit", false, [], null, this["octdigit"]);
                });
            });
        }) && (v = this._getIntermediate(), true) && this._exec(parseInt(v, 8));
    });
};

Condition.prototype["hexinteger"] = function $hexinteger() {
    var v;
    return this._seq(/^(0x)/i) && this._many(function() {
        return this._atomic(function() {
            return this._rule("hexdigit", false, [], null, this["hexdigit"]);
        });
    }) && (v = this._getIntermediate(), true) && this._exec(parseInt(v.join(""), 16));
};

Condition.prototype["bininteger"] = function $bininteger() {
    var v;
    return this._seq(/^(0b)/i) && this._many(function() {
        return this._atomic(function() {
            return this._rule("bindigit", false, [], null, this["bindigit"]);
        });
    }) && (v = this._getIntermediate(), true) && this._exec(parseInt(v.join(""), 2));
};

Condition.prototype["nonzerodigit"] = function $nonzerodigit() {
    return this._seq(/^([1-9])/);
};

Condition.prototype["octdigit"] = function $octdigit() {
    return this._seq(/^([0-7])/);
};

Condition.prototype["bindigit"] = function $bindigit() {
    return this._seq(/^([01])/);
};

Condition.prototype["hexdigit"] = function $hexdigit() {
    return this._atomic(function() {
        return this._rule("digit", false, [], null, this["digit"]);
    }) || this._seq(/^([a-f])/i);
};

Condition.prototype["floatnumber"] = function $floatnumber() {
    var f;
    return (this._atomic(function() {
        return this._rule("exponentfloat", false, [], null, this["exponentfloat"]);
    }) || this._atomic(function() {
        return this._rule("pointfloat", false, [], null, this["pointfloat"]);
    })) && (f = this._getIntermediate(), true) && this._exec(parseFloat(f));
};

Condition.prototype["pointfloat"] = function $pointfloat() {
    return this._atomic(function() {
        var x, y;
        return this._optional(function() {
            return this._rule("intpart", false, [], null, this["intpart"]);
        }) && (x = this._getIntermediate(), true) && this._rule("fraction", false, [], null, this["fraction"]) && (y = this._getIntermediate(), 
        true) && this._exec(x + "." + y);
    }) || this._atomic(function() {
        var x;
        return this._rule("intpart", false, [], null, this["intpart"]) && (x = this._getIntermediate(), 
        true) && this._match(".") && this._exec(x + ".0");
    });
};

Condition.prototype["exponentfloat"] = function $exponentfloat() {
    var f, e;
    return (this._atomic(function() {
        return this._rule("pointfloat", false, [], null, this["pointfloat"]);
    }) || this._atomic(function() {
        return this._rule("intpart", false, [], null, this["intpart"]);
    })) && (f = this._getIntermediate(), true) && this._rule("exponent", false, [], null, this["exponent"]) && (e = this._getIntermediate(), 
    true) && this._exec(f + "e" + e);
};

Condition.prototype["intpart"] = function $intpart() {
    var v;
    return this._many(function() {
        return this._atomic(function() {
            return this._rule("digit", false, [], null, this["digit"]);
        });
    }) && (v = this._getIntermediate(), true) && this._exec(v.join(""));
};

Condition.prototype["fraction"] = function $fraction() {
    var y;
    return this._match(".") && this._many(function() {
        return this._atomic(function() {
            return this._rule("digit", false, [], null, this["digit"]);
        });
    }) && (y = this._getIntermediate(), true) && this._exec(y.join(""));
};

Condition.prototype["exponent"] = function $exponent() {
    var s, e;
    return (this._match("e") || this._match("E")) && this._optional(function() {
        return this._match("+") || this._match("-");
    }) && (s = this._getIntermediate(), true) && this._many(function() {
        return this._atomic(function() {
            return this._rule("digit", false, [], null, this["digit"]);
        });
    }) && (e = this._getIntermediate(), true) && this._exec((s ? s : "") + e.join(""));
};

Condition.prototype["operator"] = function $operator() {
    return this._match("+") || this._match("-") || this._match("*") || this._seq("**") || this._seq("//") || this._match("/") || this._match("%") || this._seq("<<") || this._seq(">>") || this._match("&") || this._match("|") || this._match("^") || this._match("~") || this._seq("<=") || this._match("<") || this._seq(">=") || this._match(">") || this._seq("==") || this._seq("!=") || this._seq("<>");
};

Condition.prototype["delimiter"] = function $delimiter() {
    return this._match("(") || this._match(")") || this._match("[") || this._match("]") || this._match("{") || this._match("}") || this._match("@") || this._match(",") || this._match(":") || this._match(".") || this._match("`") || this._match("=") || this._match(";") || this._seq("+=") || this._seq("-=") || this._seq("*=") || this._seq("/=") || this._seq("//=") || this._seq("%=") || this._seq("&=") || this._seq("|=") || this._seq("^=") || this._seq(">>=") || this._seq("<<=") || this._seq("**=");
};

Condition.prototype["atom"] = function $atom() {
    return this._atomic(function() {
        return this._rule("identifier", false, [], null, this["identifier"]);
    }) || this._atomic(function() {
        return this._rule("literal", false, [], null, this["literal"]);
    }) || this._atomic(function() {
        return this._rule("enclosure", false, [], null, this["enclosure"]);
    });
};

Condition.prototype["enclosure"] = function $enclosure() {
    return this._atomic(function() {
        return this._rule("parenth_form", false, [], null, this["parenth_form"]);
    }) || this._atomic(function() {
        return this._rule("list_display", false, [], null, this["list_display"]);
    }) || this._atomic(function() {
        return this._rule("generator_expression", false, [], null, this["generator_expression"]);
    }) || this._atomic(function() {
        return this._rule("dict_display", false, [], null, this["dict_display"]);
    }) || this._atomic(function() {
        return this._rule("set_display", false, [], null, this["set_display"]);
    }) || this._atomic(function() {
        return this._rule("string_conversion", false, [], null, this["string_conversion"]);
    }) || this._atomic(function() {
        return this._rule("yield_atom", false, [], null, this["yield_atom"]);
    });
};

Condition.prototype["identifier"] = function $identifier() {
    var v;
    return this._list(function() {
        return (this._atomic(function() {
            return this._rule("letter", false, [], null, this["letter"]);
        }) || this._match("_")) && this._any(function() {
            return this._atomic(function() {
                return this._atomic(function() {
                    return this._rule("letter", false, [], null, this["letter"]);
                }) || this._atomic(function() {
                    return this._rule("digit", false, [], null, this["digit"]);
                }) || this._match("_");
            });
        });
    }, true) && (v = this._getIntermediate(), true) && this._exec(this._options.variables[v]);
};

Condition.prototype["parenth_form"] = function $parenth_form() {
    return this._match("(") && this._optional(function() {
        return this._rule("expression_list", false, [], null, this["expression_list"]);
    }) && this._match(")");
};

Condition.prototype["list_display"] = function $list_display() {
    return this._match("[") && this._optional(function() {
        return this._atomic(function() {
            return this._rule("expression_list", false, [], null, this["expression_list"]);
        }) || this._atomic(function() {
            return this._rule("list_comprehension", false, [], null, this["list_comprehension"]);
        });
    }) && this._match("]");
};

Condition.prototype["list_comprehension"] = function $list_comprehension() {
    return this._rule("expression", false, [], null, this["expression"]) && this._rule("list_for", false, [], null, this["list_for"]);
};

Condition.prototype["list_for"] = function $list_for() {
    return this._seq("for") && this._rule("target_list", false, [], null, this["target_list"]) && this._seq("in") && this._rule("old_expression_list", false, [], null, this["old_expression_list"]) && this._optional(function() {
        return this._rule("list_iter", false, [], null, this["list_iter"]);
    });
};

Condition.prototype["old_expression_list"] = function $old_expression_list() {
    return this._rule("old_expression", false, [], null, this["old_expression"]) && this._optional(function() {
        return this._many(function() {
            return this._atomic(function() {
                return this._match(",") && this._rule("old_expression", false, [], null, this["old_expression"]);
            });
        }) && this._optional(function() {
            return this._match(",");
        });
    });
};

Condition.prototype["old_expression"] = function $old_expression() {
    return this._atomic(function() {
        return this._rule("or_test", false, [], null, this["or_test"]);
    }) || this._atomic(function() {
        return this._rule("old_lambda_form", false, [], null, this["old_lambda_form"]);
    });
};

Condition.prototype["list_iter"] = function $list_iter() {
    return this._atomic(function() {
        return this._rule("list_for", false, [], null, this["list_for"]);
    }) || this._atomic(function() {
        return this._rule("list_if", false, [], null, this["list_if"]);
    });
};

Condition.prototype["list_if"] = function $list_if() {
    return this._seq("if") && this._rule("old_expression", false, [], null, this["old_expression"]) && this._optional(function() {
        return this._rule("list_iter", false, [], null, this["list_iter"]);
    });
};

Condition.prototype["comprehension"] = function $comprehension() {
    return this._rule("expression", false, [], null, this["expression"]) && this._rule("comp_for", false, [], null, this["comp_for"]);
};

Condition.prototype["comp_for"] = function $comp_for() {
    return this._seq("for") && this._rule("target_list", false, [], null, this["target_list"]) && this._seq("in") && this._rule("or_test", false, [], null, this["or_test"]) && this._optional(function() {
        return this._rule("comp_iter", false, [], null, this["comp_iter"]);
    });
};

Condition.prototype["comp_iter"] = function $comp_iter() {
    return this._atomic(function() {
        return this._rule("comp_for", false, [], null, this["comp_for"]);
    }) || this._atomic(function() {
        return this._rule("comp_if", false, [], null, this["comp_if"]);
    });
};

Condition.prototype["comp_if"] = function $comp_if() {
    return this._seq("if") && this._rule("expression_nocond", false, [], null, this["expression_nocond"]) && this._optional(function() {
        return this._rule("comp_iter", false, [], null, this["comp_iter"]);
    });
};

Condition.prototype["generator_expression"] = function $generator_expression() {
    return this._match("(") && this._rule("expression", false, [], null, this["expression"]) && this._rule("comp_for", false, [], null, this["comp_for"]) && this._match(")");
};

Condition.prototype["dict_display"] = function $dict_display() {
    return this._match("{") && this._optional(function() {
        return this._atomic(function() {
            return this._rule("key_datum_list", false, [], null, this["key_datum_list"]);
        }) || this._atomic(function() {
            return this._rule("dict_comprehension", false, [], null, this["dict_comprehension"]);
        });
    }) && this._match("}");
};

Condition.prototype["key_datum_list"] = function $key_datum_list() {
    return this._rule("key_datum", false, [], null, this["key_datum"]) && this._any(function() {
        return this._atomic(function() {
            return this._match(",") && this._rule("key_datum", false, [], null, this["key_datum"]);
        });
    }) && this._optional(function() {
        return this._match(",");
    });
};

Condition.prototype["key_datum"] = function $key_datum() {
    return this._rule("expression", false, [], null, this["expression"]) && this._match(":") && this._rule("expression", false, [], null, this["expression"]);
};

Condition.prototype["dict_comprehension"] = function $dict_comprehension() {
    return this._rule("expression", false, [], null, this["expression"]) && this._match(":") && this._rule("expression", false, [], null, this["expression"]) && this._rule("comp_for", false, [], null, this["comp_for"]);
};

Condition.prototype["set_display"] = function $set_display() {
    return this._match("{") && (this._atomic(function() {
        return this._rule("expression_list", false, [], null, this["expression_list"]);
    }) || this._atomic(function() {
        return this._rule("comprehension", false, [], null, this["comprehension"]);
    })) && this._match("}");
};

Condition.prototype["string_conversion"] = function $string_conversion() {
    return this._match("`") && this._rule("expression_list", false, [], null, this["expression_list"]) && this._match("`");
};

Condition.prototype["yield_atom"] = function $yield_atom() {
    return this._match("(") && this._rule("yield_expression", false, [], null, this["yield_expression"]) && this._match(")");
};

Condition.prototype["yield_expression"] = function $yield_expression() {
    return this._seq("yield") && this._optional(function() {
        return this._rule("expression_list", false, [], null, this["expression_list"]);
    });
};

Condition.prototype["primary"] = function $primary() {
    return this._atomic(function() {
        return this._optional(function() {
            return this._rule("spaces", true, [], null, this["spaces"]);
        }) && this._rule("atom", false, [], null, this["atom"]);
    }) || this._atomic(function() {
        return this._rule("attributeref", false, [], null, this["attributeref"]);
    }) || this._atomic(function() {
        return this._rule("subscription", false, [], null, this["subscription"]);
    }) || this._atomic(function() {
        return this._rule("slicing", false, [], null, this["slicing"]);
    }) || this._atomic(function() {
        return this._rule("call", false, [], null, this["call"]);
    });
};

Condition.prototype["attributeref"] = function $attributeref() {
    return this._rule("primary", false, [], null, this["primary"]) && this._match(".") && this._rule("identifier", false, [], null, this["identifier"]);
};

Condition.prototype["subscription"] = function $subscription() {
    return this._rule("primary", false, [], null, this["primary"]) && this._match("[") && this._rule("expression_list", false, [], null, this["expression_list"]) && this._match("]");
};

Condition.prototype["slicing"] = function $slicing() {
    return this._atomic(function() {
        return this._rule("simple_slicing", false, [], null, this["simple_slicing"]);
    }) || this._atomic(function() {
        return this._rule("extended_slicing", false, [], null, this["extended_slicing"]);
    });
};

Condition.prototype["simple_slicing"] = function $simple_slicing() {
    return this._rule("primary", false, [], null, this["primary"]) && this._match("[") && this._rule("short_slice", false, [], null, this["short_slice"]) && this._match("]");
};

Condition.prototype["extended_slicing"] = function $extended_slicing() {
    return this._rule("primary", false, [], null, this["primary"]) && this._match("[") && this._rule("slice_list", false, [], null, this["slice_list"]) && this._match("]");
};

Condition.prototype["slice_list"] = function $slice_list() {
    return this._rule("slice_item", false, [], null, this["slice_item"]) && this._any(function() {
        return this._atomic(function() {
            return this._match(",") && this._rule("slice_item", false, [], null, this["slice_item"]);
        });
    }) && this._optional(function() {
        return this._match(",");
    });
};

Condition.prototype["slice_item"] = function $slice_item() {
    return this._atomic(function() {
        return this._rule("expression", false, [], null, this["expression"]);
    }) || this._atomic(function() {
        return this._rule("proper_slice", false, [], null, this["proper_slice"]);
    }) || this._atomic(function() {
        return this._rule("ellipsis", false, [], null, this["ellipsis"]);
    });
};

Condition.prototype["proper_slice"] = function $proper_slice() {
    return this._atomic(function() {
        return this._rule("short_slice", false, [], null, this["short_slice"]);
    }) || this._atomic(function() {
        return this._rule("long_slice", false, [], null, this["long_slice"]);
    });
};

Condition.prototype["short_slice"] = function $short_slice() {
    return this._optional(function() {
        return this._rule("lower_bound", false, [], null, this["lower_bound"]);
    }) && this._match(":") && this._optional(function() {
        return this._rule("upper_bound", false, [], null, this["upper_bound"]);
    });
};

Condition.prototype["long_slice"] = function $long_slice() {
    return this._rule("short_slice", false, [], null, this["short_slice"]) && this._match(":") && this._optional(function() {
        return this._rule("stride", false, [], null, this["stride"]);
    });
};

Condition.prototype["lower_bound"] = function $lower_bound() {
    return this._rule("expression", false, [], null, this["expression"]);
};

Condition.prototype["upper_bound"] = function $upper_bound() {
    return this._rule("expression", false, [], null, this["expression"]);
};

Condition.prototype["stride"] = function $stride() {
    return this._rule("expression", false, [], null, this["expression"]);
};

Condition.prototype["ellipsis"] = function $ellipsis() {
    return this._seq("...");
};

Condition.prototype["call"] = function $call() {
    return this._rule("primary", false, [], null, this["primary"]) && this._match("(") && this._list(function() {
        return this._rule("argument_list", false, [], null, this["argument_list"]) && this._optional(function() {
            return this._match(",");
        }) && this._skip() && this._rule("expression", false, [], null, this["expression"]) && this._rule("genexpr_for", false, [], null, this["genexpr_for"]);
    }) && this._match(")");
};

Condition.prototype["argument_list"] = function $argument_list() {
    return this._rule("positional_arguments", false, [], null, this["positional_arguments"]) && this._optional(function() {
        return this._match(",") && this._rule("keyword_arguments", false, [], null, this["keyword_arguments"]);
    }) && this._optional(function() {
        return this._match(",") && this._match("*") && this._rule("expression", false, [], null, this["expression"]);
    }) && this._optional(function() {
        return this._match(",") && this._rule("keyword_arguments", false, [], null, this["keyword_arguments"]);
    }) && this._optional(function() {
        return this._match(",") && this._seq("**") && this._rule("expression", false, [], null, this["expression"]);
    });
};

Condition.prototype[""] = function $() {
    return this._skip() && this._rule("keyword_arguments", false, [], null, this["keyword_arguments"]) && this._optional(function() {
        return this._match(",") && this._match("*") && this._rule("expression", false, [], null, this["expression"]);
    }) && this._optional(function() {
        return this._match(",") && this._seq("**") && this._rule("expression", false, [], null, this["expression"]);
    }) && this._skip() && this._match("*") && this._rule("expression", false, [], null, this["expression"]) && this._optional(function() {
        return this._match(",") && this._match("*") && this._rule("expression", false, [], null, this["expression"]);
    }) && this._optional(function() {
        return this._match(",") && this._seq("**") && this._rule("expression", false, [], null, this["expression"]);
    }) && this._skip() && this._seq("**") && this._rule("expression", false, [], null, this["expression"]);
};

Condition.prototype["positional_arguments"] = function $positional_arguments() {
    return this._rule("expression", false, [], null, this["expression"]) && this._any(function() {
        return this._atomic(function() {
            return this._match(",") && this._rule("expression", false, [], null, this["expression"]);
        });
    });
};

Condition.prototype["keyword_arguments"] = function $keyword_arguments() {
    return this._rule("keyword_item", false, [], null, this["keyword_item"]) && this._any(function() {
        return this._atomic(function() {
            return this._match(",") && this._rule("keyword_item", false, [], null, this["keyword_item"]);
        });
    });
};

Condition.prototype["keyword_item"] = function $keyword_item() {
    return this._rule("identifier", false, [], null, this["identifier"]) && this._rule("token", true, [ "=" ], null, this["token"]) && this._rule("expression", false, [], null, this["expression"]);
};

Condition.prototype["power"] = function $power() {
    return this._atomic(function() {
        var l, r;
        return this._rule("primary", false, [], null, this["primary"]) && (l = this._getIntermediate(), 
        true) && this._rule("token", true, [ "**" ], null, this["token"]) && this._rule("u_expr", false, [], null, this["u_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "**", l, r ]);
    }) || this._atomic(function() {
        return this._rule("primary", false, [], null, this["primary"]);
    });
};

Condition.prototype["u_expr"] = function $u_expr() {
    return this._atomic(function() {
        var r;
        return this._rule("token", true, [ "-" ], null, this["token"]) && this._rule("u_expr", false, [], null, this["u_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec(-r);
    }) || this._atomic(function() {
        var r;
        return this._rule("token", true, [ "+" ], null, this["token"]) && this._rule("u_expr", false, [], null, this["u_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec(r);
    }) || this._atomic(function() {
        var r;
        return this._rule("token", true, [ "~" ], null, this["token"]) && this._rule("u_expr", false, [], null, this["u_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec(~r);
    }) || this._atomic(function() {
        return this._rule("power", false, [], null, this["power"]);
    });
};

Condition.prototype["m_expr"] = function $m_expr() {
    return this._atomic(function() {
        var l, r;
        return this._rule("m_expr", false, [], null, this["m_expr"]) && (l = this._getIntermediate(), 
        true) && this._rule("token", true, [ "*" ], null, this["token"]) && this._rule("u_expr", false, [], null, this["u_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "*", l, r ]);
    }) || this._atomic(function() {
        var l, r;
        return this._rule("m_expr", false, [], null, this["m_expr"]) && (l = this._getIntermediate(), 
        true) && this._rule("token", true, [ "//" ], null, this["token"]) && this._rule("u_expr", false, [], null, this["u_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "//", l, r ]);
    }) || this._atomic(function() {
        var l, r;
        return this._rule("m_expr", false, [], null, this["m_expr"]) && (l = this._getIntermediate(), 
        true) && this._rule("token", true, [ "/" ], null, this["token"]) && this._rule("u_expr", false, [], null, this["u_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "/", l, r ]);
    }) || this._atomic(function() {
        var l, r;
        return this._rule("m_expr", false, [], null, this["m_expr"]) && (l = this._getIntermediate(), 
        true) && this._rule("token", true, [ "%" ], null, this["token"]) && this._rule("u_expr", false, [], null, this["u_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "%", l, r ]);
    }) || this._atomic(function() {
        return this._rule("u_expr", false, [], null, this["u_expr"]);
    });
};

Condition.prototype["a_expr"] = function $a_expr() {
    return this._atomic(function() {
        var l, r;
        return this._rule("a_expr", false, [], null, this["a_expr"]) && (l = this._getIntermediate(), 
        true) && this._rule("token", true, [ "+" ], null, this["token"]) && this._rule("m_expr", false, [], null, this["m_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "+", l, r ]);
    }) || this._atomic(function() {
        var l, r;
        return this._rule("a_expr", false, [], null, this["a_expr"]) && (l = this._getIntermediate(), 
        true) && this._rule("token", true, [ "-" ], null, this["token"]) && this._rule("m_expr", false, [], null, this["m_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "-", l, r ]);
    }) || this._atomic(function() {
        return this._rule("m_expr", false, [], null, this["m_expr"]);
    });
};

Condition.prototype["shift_expr"] = function $shift_expr() {
    return this._atomic(function() {
        var l, op, r;
        return this._rule("shift_expr", false, [], null, this["shift_expr"]) && (l = this._getIntermediate(), 
        true) && (this._atomic(function() {
            return this._rule("token", true, [ "<<" ], null, this["token"]);
        }) || this._atomic(function() {
            return this._rule("token", true, [ ">>" ], null, this["token"]);
        })) && (op = this._getIntermediate(), true) && this._rule("a_expr", false, [], null, this["a_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec([ op, l, r ]);
    }) || this._atomic(function() {
        return this._rule("a_expr", false, [], null, this["a_expr"]);
    });
};

Condition.prototype["and_expr"] = function $and_expr() {
    return this._atomic(function() {
        var l, r;
        return this._rule("and_expr", false, [], null, this["and_expr"]) && (l = this._getIntermediate(), 
        true) && this._rule("token", true, [ "&" ], null, this["token"]) && this._rule("shift_expr", false, [], null, this["shift_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "&", l, r ]);
    }) || this._atomic(function() {
        return this._rule("shift_expr", false, [], null, this["shift_expr"]);
    });
};

Condition.prototype["xor_expr"] = function $xor_expr() {
    return this._atomic(function() {
        var l, r;
        return this._rule("xor_expr", false, [], null, this["xor_expr"]) && (l = this._getIntermediate(), 
        true) && this._rule("token", true, [ "^" ], null, this["token"]) && this._rule("and_expr", false, [], null, this["and_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "^", l, r ]);
    }) || this._atomic(function() {
        return this._rule("and_expr", false, [], null, this["and_expr"]);
    });
};

Condition.prototype["or_expr"] = function $or_expr() {
    return this._atomic(function() {
        var l, r;
        return this._rule("or_expr", false, [], null, this["or_expr"]) && (l = this._getIntermediate(), 
        true) && this._rule("token", true, [ "|" ], null, this["token"]) && this._rule("xor_expr", false, [], null, this["xor_expr"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "|", l, r ]);
    }) || this._atomic(function() {
        return this._rule("xor_expr", false, [], null, this["xor_expr"]);
    });
};

Condition.prototype["comparison"] = function $comparison() {
    return this._atomic(function() {
        var l, op, r;
        return this._rule("or_expr", false, [], null, this["or_expr"]) && (l = this._getIntermediate(), 
        true) && this._rule("comp_operator", false, [], null, this["comp_operator"]) && (op = this._getIntermediate(), 
        true) && this._rule("comparison", false, [], null, this["comparison"]) && (r = this._getIntermediate(), 
        true) && this._exec([ op, l, r ]);
    }) || this._atomic(function() {
        return this._rule("or_expr", false, [], null, this["or_expr"]);
    });
};

Condition.prototype["comp_operator"] = function $comp_operator() {
    return this._atomic(function() {
        return this._rule("token", true, [ "<=" ], null, this["token"]);
    }) || this._atomic(function() {
        return this._rule("token", true, [ ">=" ], null, this["token"]);
    }) || this._atomic(function() {
        return this._rule("token", true, [ "<" ], null, this["token"]);
    }) || this._atomic(function() {
        return this._rule("token", true, [ ">" ], null, this["token"]);
    }) || this._atomic(function() {
        return this._rule("token", true, [ "==" ], null, this["token"]);
    }) || this._atomic(function() {
        return this._rule("token", true, [ "<>" ], null, this["token"]);
    }) || this._atomic(function() {
        return this._rule("token", true, [ "!=" ], null, this["token"]);
    }) || this._atomic(function() {
        return this._rule("token", true, [ "is" ], null, this["token"]) && this._optional(function() {
            return this._rule("token", true, [ "not" ], null, this["token"]);
        });
    }) || this._atomic(function() {
        return this._optional(function() {
            return this._rule("token", true, [ "not" ], null, this["token"]);
        }) && this._rule("token", true, [ "in" ], null, this["token"]);
    });
};

Condition.prototype["or_test"] = function $or_test() {
    return this._atomic(function() {
        return this._rule("and_test", false, [], null, this["and_test"]);
    }) || this._atomic(function() {
        var l, r;
        return this._rule("or_test", false, [], null, this["or_test"]) && (l = this._getIntermediate(), 
        true) && this._rule("token", true, [ "or" ], null, this["token"]) && this._rule("and_test", false, [], null, this["and_test"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "or", l, r ]);
    });
};

Condition.prototype["and_test"] = function $and_test() {
    return this._atomic(function() {
        return this._rule("not_test", false, [], null, this["not_test"]);
    }) || this._atomic(function() {
        var r;
        return this._rule("and_test", false, [], null, this["and_test"]) && (r = this._getIntermediate(), 
        true) && this._rule("token", true, [ "and" ], null, this["token"]) && this._rule("not_test", false, [], null, this["not_test"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "and", l, r ]);
    });
};

Condition.prototype["not_test"] = function $not_test() {
    return this._atomic(function() {
        return this._rule("comparison", false, [], null, this["comparison"]);
    }) || this._atomic(function() {
        var r;
        return this._rule("token", true, [ "not" ], null, this["token"]) && this._rule("not_test", false, [], null, this["not_test"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "not", r ]);
    });
};

Condition.prototype["conditional_expression"] = function $conditional_expression() {
    return this._atomic(function() {
        var l, c, r;
        return this._rule("or_test", false, [], null, this["or_test"]) && (l = this._getIntermediate(), 
        true) && this._rule("token", true, [ "if" ], null, this["token"]) && this._rule("or_test", false, [], null, this["or_test"]) && (c = this._getIntermediate(), 
        true) && this._rule("token", true, [ "else" ], null, this["token"]) && this._rule("expression", false, [], null, this["expression"]) && (r = this._getIntermediate(), 
        true) && this._exec([ "if", c, l, r ]);
    }) || this._atomic(function() {
        return this._rule("or_test", false, [], null, this["or_test"]);
    });
};

Condition.prototype["expression"] = function $expression() {
    return this._atomic(function() {
        return this._rule("conditional_expression", false, [], null, this["conditional_expression"]);
    }) || this._atomic(function() {
        return this._rule("lambda_form", false, [], null, this["lambda_form"]);
    });
};

Condition.prototype["lambda_form"] = function $lambda_form() {
    return this._rule("token", true, [ "lambda" ], null, this["token"]) && this._optional(function() {
        return this._rule("parameter_list", false, [], null, this["parameter_list"]);
    }) && this._match(":") && this._rule("expression", false, [], null, this["expression"]);
};

Condition.prototype["old_lambda_form"] = function $old_lambda_form() {
    return this._rule("token", true, [ "lambda" ], null, this["token"]) && this._optional(function() {
        return this._rule("parameter_list", false, [], null, this["parameter_list"]);
    }) && this._match(":") && this._rule("old_expression", false, [], null, this["old_expression"]);
};

Condition.prototype["expression_list"] = function $expression_list() {
    return this._rule("expression", false, [], null, this["expression"]) && this._any(function() {
        return this._atomic(function() {
            return this._rule("token", true, [ "," ], null, this["token"]) && this._rule("expression", false, [], null, this["expression"]);
        });
    }) && this._optional(function() {
        return this._match(",");
    });
};

Condition.prototype["token"] = function $token() {
    var t;
    return this._optional(function() {
        return this._rule("spaces", true, [], null, this["spaces"]);
    }) && (this._atomic(function() {
        return this._rule("keyword", true, [], null, this["keyword"]);
    }) || this._atomic(function() {
        return this._rule("literal", true, [], null, this["literal"]);
    }) || this._atomic(function() {
        return this._rule("identifier", true, [], null, this["identifier"]);
    }) || this._atomic(function() {
        return this._rule("operator", true, [], null, this["operator"]);
    }) || this._atomic(function() {
        return this._rule("delimiter", true, [], null, this["delimiter"]);
    })) && (t = this._getIntermediate(), true) && this._exec([ t, t ]);
};

Condition.prototype["expr"] = function $expr() {
    var e;
    return this._rule("expression", false, [], null, this["expression"]) && (e = this._getIntermediate(), 
    true) && this._optional(function() {
        return this._rule("spaces", true, [], null, this["spaces"]);
    }) && this._rule("end", false, [], null, this["end"]) && this._exec(e);
};

var Evaluator = function Evaluator(source, opts) {
    AbstractGrammar.call(this, source, opts);
};

Evaluator.grammarName = "Evaluator";

Evaluator.match = AbstractGrammar.match;

Evaluator.matchAll = AbstractGrammar.matchAll;

exports.Evaluator = Evaluator;

require("util").inherits(Evaluator, AbstractGrammar);

Evaluator.prototype["interp"] = function $interp() {
    return this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match(">") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x > y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("<") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x < y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match(">=") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x >= y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("<=") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x <= y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("!=") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x != y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("<>") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x != y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("==") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x == y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("is") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x == y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("is not") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x != y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("+") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x + y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("-") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x - y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("*") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x * y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("//") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(Math.floor(x / y));
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("/") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x / y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("%") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x % y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("&") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x & y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("|") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(function() {
            return x | y;
        }.call(this));
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("^") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x ^ y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match("<<") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x << y);
    }) || this._atomic(function() {
        var x, y;
        return this._list(function() {
            return this._match(">>") && this._rule("interp", false, [], null, this["interp"]) && (x = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (y = this._getIntermediate(), 
            true);
        }) && this._exec(x >> y);
    }) || this._atomic(function() {
        var c, t, e;
        return this._list(function() {
            return this._match("if") && this._rule("interp", false, [], null, this["interp"]) && (c = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (t = this._getIntermediate(), 
            true) && this._rule("interp", false, [], null, this["interp"]) && (e = this._getIntermediate(), 
            true);
        }) && this._exec(c ? t : e);
    }) || this._atomic(function() {
        var x;
        return this._skip() && (x = this._getIntermediate(), true) && this._exec(x);
    });
};

function unescape(c) {
    switch (c) {
      case "\n":
        return "";

      case "\\":
        return "\\";

      case '"':
        return '"';

      case "'":
        return "'";

      case "a":
        return "";

      case "b":
        return "\b";

      case "f":
        return "";

      case "n":
        return "\n";

      case "N":
        throw new Error("not supported");

      case "r":
        return "\r";

      case "t":
        return "	";

      case "v":
        return "";

      default:
        return "\\" + c;
    }
}

module.exports = function(str, variables) {
    var tree = Condition.matchAll(str, "expr", {
        variables: variables
    });
    return Evaluator.match(tree, "interp");
};

module.exports.evaluator = Evaluator;

module.exports.parser = Condition;