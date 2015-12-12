var StateMachine = require("../lib/statemachines");

var nfa = new StateMachine.Nondeterministic([
    { "" : [1, 7] },
    { "" : [2, 4] },
    { "a": [3] },
    { "" : [6] },
    { "b": [5] },
    { "" : [6] },
    { "" : [1, 7] },
    { "a": [8] },
    { "b": [9] },
    { "b": [10] },
    { }
], [10]);

if (nfa.test("abb") &&
    nfa.test("aabb") &&
    nfa.test("babb") &&
    nfa.test("aaabb") &&
    nfa.test("ababb") &&
    !nfa.test("abba") &&
    !nfa.test("cabb")) console.log("Passed all tests.");
else {
    console.error("Failed test(s).");
    process.exit(1);
}
