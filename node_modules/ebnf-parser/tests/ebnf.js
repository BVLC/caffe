var assert = require("assert"),
    ebnf = require("../ebnf-transform");
var Parser = require('jison').Parser;

function testParse(top, strings) {
    return function() {
        var grammar = {
            "lex": {
                "rules": [
                    ["\\s+", ''],
                    ["[A-Za-z]+", "return 'word';"],
                    [",", "return ',';"],
                    ["$", "return 'EOF';"]
                ]
            },
            "start": "top",
            "bnf": ebnf.transform({"top": [top]})
        };
        strings = (typeof(strings) === 'string' ? [strings] : strings);
        strings.forEach(function(string) {
            assert.ok(new Parser(grammar).parse(string));
        });
    };
}

function testBadParse(top, strings) {
    return function() {
        var grammar = {
            "lex": {
                "rules": [
                    ["\\s+", ''],
                    ["[A-Za-z]+", "return 'word';"],
                    [",", "return ',';"],
                    ["$", "return 'EOF';"]
                ]
            },
            "start": "top",
            "ebnf": {"top": [top]}
        };
        strings = (typeof(strings) === 'string' ? [strings] : strings);
        strings.forEach(function(string) {
            assert.throws(function () {new Parser(grammar).parse(string);})
        });
    };
}

function testAlias(top, obj, str) {
    return function() {
        var grammar = {
            "lex": {
                "rules": [
                    ["\\s+", ''],
                    ["[A-Za-z]+", "return 'word';"],
                    [",", "return ',';"],
                    ["$", "return 'EOF';"]
                ]
            },
            "start": "top",
            "bnf": ebnf.transform({"top": [top]})
        };
        assert.deepEqual(grammar['bnf'], obj);
        assert.ok(new Parser(grammar).parse(str));
    };
}

var tests = {
    "test idempotent transform": function() {
        var first = {
            "nodelist": [["", "$$ = [];"], ["nodelist node", "$1.push($2);"]]
        };
        var second = ebnf.transform(JSON.parse(JSON.stringify(first)));
        assert.deepEqual(second, first);
    },
    "test repeat (*) on empty string": testParse("word* EOF", ""),
    "test repeat (*) on single word": testParse("word* EOF", "oneword"),
    "test repeat (*) on multiple words": testParse("word* EOF", "multiple words"),
    "test repeat (+) on empty string": testBadParse("word+ EOF", ""),
    "test repeat (+) on single word": testParse("word+ EOF", "oneword"),
    "test repeat (+) on multiple words": testParse("word+ EOF", "multiple words"),
    "test option (?) on empty string": testParse("word? EOF", ""),
    "test option (?) on single word": testParse("word? EOF", "oneword"),
    "test group () on simple phrase": testParse("(word word) EOF", "two words"),
    "test group () with multiple options on first option": testParse("((word word) | word) EOF", "hi there"),
    "test group () with multiple options on second option": testParse("((word word) | word) EOF", "hi"),
    "test complex expression ( *, ?, () )": testParse("(word (',' word)*)? EOF ", ["", "hi", "hi, there"]),
    "test named repeat (*)": testAlias("word*[bob] EOF",
        { top: [ 'bob EOF' ],
        bob: [ [ '', '$$ = [];' ], [ 'bob word', '$1.push($2);' ] ] }, "word"),
    "test named repeat (+)": testAlias("word+[bob] EOF",
        { top: [ 'bob EOF' ],
        bob: [ [ 'word', '$$ = [$1];' ], [ 'bob word', '$1.push($2);' ] ] }, "wordy word"),
    "test named group ()": testAlias("word[alice] (',' word)*[bob] EOF",
        {"top":["word[alice] bob EOF"],"bob":[["","$$ = [];"],["bob , word","$1.push($2);"]]},
        "one, two"),
    "test named option (?)": testAlias("word[alex] word?[bob] EOF", { top: [ 'word[alex] bob EOF' ], bob: [ '', 'word' ] }, "oneor two"),
    "test named complex expression (())": testAlias("word[alpha] (word[alex] (word[bob] word[carol] ',')+[david] word ',')*[enoch] EOF",
        {"top":["word[alpha] enoch EOF"],"david":[["word[bob] word[carol] ,","$$ = [$1];"],["david word[bob] word[carol] ,","$1.push($2);"]],
        "enoch":[["","$$ = [];"],["enoch word[alex] david word ,","$1.push($2);"]]},
        "one two three four, five,"
    )
};

for (var test in tests) {
    exports[test] = tests[test];
}
