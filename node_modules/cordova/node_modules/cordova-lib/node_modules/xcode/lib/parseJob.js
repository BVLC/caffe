// parsing is slow and blocking right now
// so we do it in a separate process
var fs = require('fs'),
    parser = require('./parser/pbxproj'),
    path = process.argv[2],
    fileContents, obj;

try {
    fileContents = fs.readFileSync(path, 'utf-8'),
    obj = parser.parse(fileContents)
    process.send(obj)
    process.exit()
} catch (e) {
    process.send(e)
    process.exit(1)
}
