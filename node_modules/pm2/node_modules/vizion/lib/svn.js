var fs = require('fs');
var async = require('async');
var exec = require('child_process').exec;

var svn = {};

svn.parse = function(folder, cb) {
  var getMeta = function(cb) {
    exec("cd '"+folder+"';LC_ALL=en_US.UTF-8 svn info", function(err, stdout, stderr) {
      if(err !== null)
        return cb(err);
      var data = {};
      data.type = 'svn';
      data.url = stdout.match(/Repository Root: ([^\n]+)/);
      if (data.url && typeof(data.url) === 'object') {
        data.url = data.url[1];
        data.branch = typeof(data.url) === 'string' ? data.url.match(/[^/]+$/) : null;
      }
      if (data.branch) data.branch = data.branch[0];
      return cb(null, data);
    });
  }

  var getRevComment = function(data, cb) {
    exec("cd '"+folder+"';LC_ALL=en_US.UTF-8 svn log -r BASE", function(err, stdout, stderr) {
      if(err !== null)
        return cb(err);
      data.revision = stdout.match(/^(r[0-9]+)\s\|/m);
      data.comment = stdout.match(/lines?\s*\n((.|\n)*)\n-{72}\n$/);
      if (data.revision) data.revision = data.revision[1];
      if (data.comment) data.comment = data.comment[1].replace(/\n/g, '');
      cb(null, data);
    });
  }

  var getDate = function(data, cb) {
    fs.stat(folder+".svn", function(err, stats) {
      if(err !== null)
        return cb(err);
      data.update_time = stats.mtime;
      return cb(null, data);
    });
  }

  async.waterfall([getMeta, getRevComment, getDate],
  function(err, data) {
    if (err !== null)
      return cb(err);
    return cb(null, data);
  });
}

svn.isUpdated = function(folder, cb) {
  var res = {};

  var getRev = function(str) {
    var matches = str.match(/Changed Rev: ([^\n]+)/);
    if (matches) matches = matches[1];
    return matches;
  }

  exec("cd '"+folder+"';LC_ALL=en_US.UTF-8 svn info", function(err, stdout, stderr) {
    if(err !== null)
      return cb(err);
    var current_rev = getRev(stdout);
    exec("cd '"+folder+"';LC_ALL=en_US.UTF-8 svn info -r HEAD", function(err, stdout, stderr) {
      if(err !== null)
        return cb(err);
      var recent_rev = getRev(stdout);
      res.is_up_to_date = (recent_rev === current_rev);
      res.new_revision = recent_rev;
      res.current_revision = current_rev;
      return cb(null, res);
    });
  });
}

svn.update = function(folder, cb) {
  var res = {};

  exec("cd '"+folder+"';LC_ALL=en_US.UTF-8 svn update", function(err, stdout, stderr) {
    if(err !== null)
      return cb(err);
    var new_rev = stdout.match(/Updated to revision ([^\.]+)/);
    if (new_rev === null)
    {
      res.success = false;
      var old_rev = stdout.match(/At revision ([^\.]+)/);
      res.current_revision = (old_rev) ? old_rev[1] : null;
    }
    else {
      res.success = true;
      res.current_revision = new_rev[1];
    }
    return cb(null, res);
  });
}

module.exports = svn;
