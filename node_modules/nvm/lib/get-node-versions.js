const http = require('http');

const sourceUrl = 'http://nodejs.org/dist/';
const pattern = /v(\d+\.\d+\.\d+)\//;

function getRegexGroup(re, n) {
  n = n || 1;
  return function (line) {
    const match = re.exec(line);
    if (!match) return null;
    return match[n];
  };
}

function versionSort(a, b) {
  const $ = parseInt;
  const aParts = a.split('.');
  const bParts = b.split('.');
  for (var i = 0; i < aParts.length; i++) {
    if ($(aParts[i]) > $(bParts[i])) return -1;
    if ($(aParts[i]) < $(bParts[i])) return 1;
  }
  return 0;
}

function getVersions(callback) {
  http.get(sourceUrl, function (res) {
    var body = '';
    res.on('data', function (chunk) { body += chunk; });

    res.on('end', function () {
      const versions = body
        .trim().split('\r\n')
        .filter(pattern.test.bind(pattern))
        .map(getRegexGroup(pattern))
        .sort(versionSort);
      return callback(versions);
    });
  });
}

module.exports = getVersions;
