if (typeof require === "function") {
    var StateMachine = require("statemachines");
    var parse = require("./parse");
    require("augment");
}

var nfa = Function.bindable(StateMachine.Nondeterministic, null);
if (typeof module === "object") module.exports = Regex;
Regex.convert = convert;
Regex.parse = parse;

function Regex(regexp) {
    var graph = parse(regexp instanceof RegExp ? regexp.source : regexp);
    var dfa = (new (nfa.apply(null, convert(graph)))).subset();
    this.test = dfa.test.bind(dfa);
}

function convert(graph) {
    var states = [graph.start];
    var transition = [];
    var cursor = 0;

    while (cursor < states.length) {
        var state = states[cursor++];
        var symbols = Object.keys(state);
        var length = symbols.length;
        var tuple = {};

        for (var i = 0; i < length; i++) {
            var symbol = symbols[i];
            var moves = state[symbol];
            var degree = moves.length;
            var transitions = [];

            for (var j = 0; j < degree; j++) {
                var move = moves[j];
                var index = states.indexOf(move);

                if (index < 0) {
                    index = states.length;
                    states.push(move);
                }

                transitions.push(index);
            }

            tuple[symbol] = transitions;
        }

        transition.push(tuple);
    }

    return [transition, [states.indexOf(graph.final)]];
}
