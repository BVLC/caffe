var Jison = require("../setup").Jison,
    assert = require("assert");

exports["test right-recursive nullable grammar"] = function () {

    var grammar = {
        tokens: [ 'x' ],
        startSymbol: "A",
        bnf: {
            "A" :[ 'x A',
                   ''      ]
        }
    };

    var gen = new Jison.Generator(grammar, {type: "slr"});
    var gen2 = new Jison.Generator(grammar, {type: "lalr"});

    assert.equal(gen.table.length, 4, "table has 4 states");
    assert.equal(gen.nullable('A'), true, "A is nullable");
    assert.equal(gen.conflicts, 0, "should have no conflict");
    assert.deepEqual(gen.table, gen2.table, "should have identical tables");
};

exports["test slr lalr lr tables are equal"] = function () {
    var grammar = {
        tokens: [ "ZERO", "PLUS"],
        startSymbol: "E",
        bnf: {
            "E" :[ "E PLUS T",
                   "T"      ],
            "T" :[ "ZERO" ]
        }
    };

    var gen = new Jison.Generator(grammar, {type: "slr"});
    var gen2 = new Jison.Generator(grammar, {type: "lalr"});
    var gen3 = new Jison.Generator(grammar, {type: "lr"});

    assert.deepEqual(gen.table, gen2.table, "slr lalr should have identical tables");
    assert.deepEqual(gen2.table, gen3.table, "lalr lr should have identical tables");
};

exports["test LL parse table"] = function () {

    var grammar = {
        tokens: [ 'x' ],
        startSymbol: "A",
        bnf: {
            "A" :[ 'x A',
                   ''      ]
        }
    };

    var gen = new Jison.Generator(grammar, {type: "ll"});

    assert.deepEqual(gen.table, {$accept:{x:[0], $end:[0]}, A:{x:[1], $end:[2]}}, "ll table has 2 states");
};

exports["test LL parse table with conflict"] = function () {

    var grammar = {
        tokens: [ 'x' ],
        startSymbol: "L",
        bnf: {
            "L" :[ 'T L T',
                   ''      ],
            "T" :[ "x" ]
        }
    };

    var gen = new Jison.Generator(grammar, {type: "ll"});
    assert.equal(gen.conflicts, 1, "should have 1 conflict");
};

exports["test Ambigous grammar"] = function () {

    var grammar = {
        tokens: [ 'x', 'y' ],
        startSymbol: "A",
        bnf: {
            "A" :[ 'A B A',
                   'x'      ],
            "B" :[ '',
                   'y'      ]
        }
    };

    var gen = new Jison.Generator(grammar, {type: "lr"});
    assert.equal(gen.conflicts, 2, "should have 2 conflict");
};

// for Minimal LR testing. Not there yet.
/*exports["test Spector grammar G1"] = function () {*/

    //var grammar = {
        //"tokens": "z d b c a",
        //"startSymbol": "S",
        //"bnf": {
            //"S" :[ "a A c",
                   //"a B d",
                   //"b A d",
                   //"b B c"],
            //"A" :[ "z" ],
            //"B" :[ "z" ]
        //}
    //};

    //var gen = new Jison.Generator(grammar, {type: "mlr", debug:true});
    //assert.strictEqual(gen.conflicts, 0, "should have no conflict");
//};

//exports["test De Remer G4"] = function () {

    //var grammar = {
        //"tokens": "z d b c a",
        //"startSymbol": "S",
        //"bnf": {
            //"S" : "a A d | b A c | b B d",
            //"A" : "e A | e",
            //"B" : "e B | e" 
        //}
    //};

    //var gen = new Jison.Generator(grammar, {type: "mlr", debug:true});
    //assert.strictEqual(gen.conflicts, 0, "should have no conflict");
/*};*/
