var jsonFile = require('jsonfile')

module.exports = {
  outputJsonSync: require('./output-json-sync'),
  outputJson: require('./output-json'),
  // aliases
  outputJSONSync: require('./output-json-sync'),
  outputJSON: require('./output-json'),
  // jsonfile exports
  readJson: jsonFile.readFile,
  readJSON: jsonFile.readFile,
  readJsonSync: jsonFile.readFileSync,
  readJSONSync: jsonFile.readFileSync,
  writeJson: jsonFile.writeFile,
  writeJSON: jsonFile.writeFile,
  writeJsonSync: jsonFile.writeFileSync,
  writeJSONSync: jsonFile.writeFileSync,
  spaces: 2 // default in fs-extra
}
