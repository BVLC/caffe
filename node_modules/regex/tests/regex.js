var Regex = require("..");

var regex = new Regex(/(a|b)*abb/);

if (regex.test("abb") &&
    regex.test("aabb") &&
    regex.test("babb") &&
    regex.test("aaabb") &&
    regex.test("ababb") &&
    !regex.test("abba") &&
    !regex.test("cabb")) console.log("Passed all tests.");
else {
    console.error("Failed test(s).");
    process.exit(1);
}
