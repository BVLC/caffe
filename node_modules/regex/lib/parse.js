if (typeof require === "function") {
    var parsexp = require("./parsexp");
    parsexp.lexer = require("./reglex");
} else parsexp.lexer = reglex;

var yy = parsexp.yy;
var parse = parsexp.parse.bind(parsexp);
if (typeof module === "object") module.exports = parse;

yy.atom = function (symbol) {
    var start = {};
    var final = {};

    start[symbol] = [final];

    return {
        start: start,
        final: final
    };
};

yy.group = function (atom) {
    var start = {};
    var final = {};

    var moves = start[""] = [atom.start, final];
    addMoves(atom.final, "", moves.slice());

    return {
        start: start,
        final: final
    };
};

yy.sequence = function (sequence, group) {
    var start = sequence.start;
    var final = group.final;

    var begin = group.start; 
    var end = sequence.final;
    addMoves(begin, "", [end]);
    addMoves(end, "", [begin]);

    return {
        start: start,
        final: final
    };
};

yy.expression = function (expression, sequence) {
    var start = {};
    var final = {};

    start[""] = [expression.start, sequence.start];
    addMoves(expression.final, "", [final]);
    addMoves(sequence.final, "", [final]);

    return {
        start: start,
        final: final
    };
};

function addMoves(state, symbol, moves) {
    var oldMoves = state[symbol];
    state[symbol] = oldMoves ? oldMoves.concat(moves) : moves;
}
