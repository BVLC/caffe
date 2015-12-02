var fs = require('fs');
var async = require('async');

module.exports = function(folder, cb) {
  if (folder[folder.length - 1] !== '/')
    folder += '/';

  async.eachSeries(['git', 'hg', 'svn'],
  function(type, callback) {
    fs.exists(folder+'.'+type, function(exists) {
      if (exists)
        return callback(type);
      else
        return callback();
    });
  },
  function(final) {
    return cb(final ? final : 'No versioning system found', folder);
  });
};
