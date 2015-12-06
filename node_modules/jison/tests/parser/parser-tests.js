exports.testAPI = require("./api");
exports.testLR0 = require("./lr0");
exports.testSLR = require("./slr");
exports.testLALR = require("./lalr");
exports.testLR1 = require("./lr1");
exports.testAST = require("./actions");
exports.testTables = require("./tables");
exports.testPrecedence = require("./precedence");
exports.testGenerator = require("./generator");
exports.testErrorLab = require("./errorlab");

if (require.main === module)
    require("test").run(exports);
