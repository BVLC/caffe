'use strict';

var isDarwin = /darwin/i.test(require('os').type());
var debug = require('debug')('ember-cli:utilities/attempt-metadata-index-file');

module.exports = function(dir) {
  var path = dir + '/.metadata_never_index';

  if (!isDarwin) {
    debug('not darwin, skipping %s (which hints to spotlight to prevent indexing)', path);
    return;
  }

  debug('creating: %s (to prevent spotlight indexing)', path);

  var fs = require('fs-extra');

  fs.mkdirsSync(dir);
  fs.writeFileSync(path);
};
