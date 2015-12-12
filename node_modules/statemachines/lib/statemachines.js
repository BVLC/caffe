if (typeof module === "object") module.exports = StateMachine;

if (typeof require === "function") {
    var SortedArray = require("sorted-array");
    require("augment");
}

var sortable = Function.bindable(SortedArray, null);

function StateMachine(transition, final) {
    this.transition = transition;
    this.final = final;
}

StateMachine.Deterministic = StateMachine.augment(function (base) {
    this.constructor = function (transition, final) {
        base.constructor.call(this, transition, final);
    };

    this.test = function (string) {
        var state = 0, index = 0;
        var length = string.length;
        var transition = this.transition;

        while (index < length) {
            state = transition[state][string.charAt(index++)];
            if (typeof state === "undefined") return false;
        }

        return this.final.indexOf(state) >= 0;
    };
});

StateMachine.Nondeterministic = StateMachine.augment(function (base) {
    this.constructor = function (transition, final) {
        base.constructor.call(this, transition, final);
    };

    this.test = function (string) {
        var index = 0;
        var length = string.length;
        var state = epsilonMoves.call(this, 0);

        while (index < length) {
            state = moveOn.apply(this, [string.charAt(index++)].concat(state));
            if (state.length) state = epsilonMoves.apply(this, state);
            else return false;
        }

        return isFinal.apply(this, state);
    };

    this.subset = function () {
        var initial = epsilonMoves.call(this, 0);
        var names = [initial.toString()];
        var states = [initial];
        var transition = [];
        var final = [];

        for (var i = 0; i < states.length; i++) {
            var state = states[i];
            var symbols = moves.apply(this, state);
            var length = symbols.length;
            var node = {};

            for (var j = 0; j < length; j++) {
                var symbol = symbols[j];
                var next = epsilonMoves.apply(this,
                    moveOn.apply(this, [symbol].concat(state)));
                var name = next.toString();
                var index = names.indexOf(name);

                if (index < 0) {
                    states.push(next);
                    index = names.length;
                    names.push(name);
                }

                node[symbol] = index;
            }

            transition.push(node);

            if (isFinal.apply(this, state)) final.push(i);
        }

        return new StateMachine.Deterministic(transition, final);
    };

    function epsilonMoves() {
        var stack = Array.from(arguments);
        var states = new (sortable.apply(null, stack));
        var transition = this.transition;

        while (stack.length) {
            var moves = transition[stack.pop()][""];

            if (moves) {
                var length = moves.length;

                for (var i = 0; i < length; i++) {
                    var move = moves[i];

                    if (states.search(move) < 0) {
                        states.insert(move);
                        stack.push(move);
                    }
                }
            }
        }

        return states.array;
    }

    function moves() {
        var transition = this.transition;
        var stack = Array.from(arguments);
        var symbols = new SortedArray;

        while (stack.length) {
            var keys = Object.keys(transition[stack.pop()]);
            var length = keys.length;

            for (var i = 0; i < length; i++) {
                var key = keys[i];

                if (symbols.search(key) < 0)
                    symbols.insert(key);
            }
        }

        return symbols.remove("").array;
    }

    function moveOn(symbol) {
        var stack = Array.from(arguments, 1);
        var transition = this.transition;
        var states = new SortedArray;

        while (stack.length) {
            var moves = transition[stack.pop()][symbol];

            if (moves) {
                var length = moves.length;

                for (var i = 0; i < length; i++) {
                    var move = moves[i];

                    if (states.search(move) < 0)
                        states.insert(move);
                }
            }
        }

        return states.array;
    }

    function isFinal() {
        var stack = Array.from(arguments);
        var final = this.final;

        while (stack.length)
            if (final.indexOf(stack.pop()) >= 0)
                return true;

        return false;
    }
});
