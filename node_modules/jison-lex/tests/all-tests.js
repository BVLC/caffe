exports.testRegExpLexer = require("./regexplexer");

if (require.main === module)
    process.exit(require("test").run(exports));
