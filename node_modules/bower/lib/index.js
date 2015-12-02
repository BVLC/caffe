var commands = require('./commands');
var pkg = require('../package.json');
var abbreviations = require('./abbreviations')(commands);

function clearRuntimeCache() {
    // Note that in edge cases, some architecture components instance's
    // in-memory cache might be skipped.
    // If that's a problem, you should create and fresh instances instead.
    var PackageRepository = require('./core/PackageRepository');
    PackageRepository.clearRuntimeCache();
}

module.exports = {
    version: pkg.version,
    commands: commands,
    config: require('./config')(),
    abbreviations: abbreviations,
    reset: clearRuntimeCache
};
