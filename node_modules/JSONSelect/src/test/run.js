/*
 * a node.js test runner that executes all conformance
 * tests and outputs results to console.
 * Process returns zero on success, non-zero on failure.
 */

const   fs = require('fs'),
      path = require('path'),
jsonselect = require('../jsonselect.js'),
       sys = require('sys');

var pathToTests = path.join(__dirname, "tests");

// a map: document nametest name -> list of se
var numTests = 0;
var numPassed = 0;
var tests = {};

function runOneSync(name, selname, p) {
    var testDocPath = path.join(p, name + ".json");
    var selDocPath = path.join(p, name + '_' +
                               selname + ".selector");
    var outputDocPath = selDocPath.replace(/selector$/, "output");

    // take `obj`, apply `sel, get `got`, is it what we `want`? 
    var obj = JSON.parse(fs.readFileSync(testDocPath));
    var want = String(fs.readFileSync(outputDocPath)).trim();
    var got = "";
    var sel = String(fs.readFileSync(selDocPath)).trim();

    try {
        jsonselect.forEach(sel, obj, function(m) {
            got += JSON.stringify(m, undefined, 4) + "\n";
        });
    } catch(e) {
        got = e.toString();
        if (want.trim() != got.trim()) throw e;
    }
    if (want.trim() != got.trim()) throw "mismatch";
}


function runTests() {
    console.log("Running Tests:"); 
    for (var l in tests) {
        for (var d in tests[l]) {
            console.log("  level " + l + " tests against \"" + d + ".json\":");
            for (var i = 0; i < tests[l][d].length; i++) {
                sys.print("    " + tests[l][d][i][0] + ": ");
                try {
                    runOneSync(d, tests[l][d][i][0], tests[l][d][i][1]);
                    numPassed++;
                    console.log("pass");
                } catch (e) {
                    console.log("fail (" + e.toString() + ")");
                }
            }
        }
    }
    console.log(numPassed + "/" + numTests + " passed");
    process.exit(numPassed == numTests ? 0 : 1);
}

// discover all tests
var pathToTests = path.join(__dirname, "tests");

fs.readdirSync(pathToTests).forEach(function(subdir) {
    var p = path.join(pathToTests, subdir);
    if (!fs.statSync(p).isDirectory()) return;
    var l = /^level_([\d+])$/.exec(subdir);
    if (!l) return;
    l = l[1];
    var files = fs.readdirSync(p);
    for (var i = 0; i < files.length; i++) {
        var f = files[i];
        var m = /^([A-Za-z]+)_(.+)\.selector$/.exec(f);
        if (m) {
            if (!tests.hasOwnProperty(l)) tests[l] = [];
            if (!tests[l].hasOwnProperty(m[1])) tests[l][m[1]] = [];
            numTests++;
            tests[l][m[1]].push([m[2], p]);
        }
    }
});
runTests();
