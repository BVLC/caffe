if (typeof require === "function") var Lexer = require("lex");

var reglex = new Lexer;

reglex.addRule(/[(|)*]/, function (lexeme) {
    return lexeme;
});

reglex.addRule(/./, function (lexeme) {
    this.yytext = lexeme;
    return "SYMBOL";
});

reglex.addRule(/$/, function () {
    return "EOF";
});

if (typeof module === "object") module.exports = reglex;
