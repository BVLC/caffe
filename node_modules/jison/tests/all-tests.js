exports.testParser = require("./parser/parser-tests");

if (require.main === module)
    require("test").run(exports);
