#!/usr/bin/env narwhal

//exports.testBNF = require("./bnf");
exports.testBNFParse = require("./bnf_parse");
exports.testEBNF = require("./ebnf");
exports.testEBNFParse = require("./ebnf_parse");

if (require.main === module)
    require("test").run(exports);
