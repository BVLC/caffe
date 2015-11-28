var pkg = require('../../package.json');

module.exports = 'node/' + process.version + ' ' + process.platform + ' ' + process.arch + ' ' + ';Bower ' + pkg.version;
