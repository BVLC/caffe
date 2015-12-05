var fs   = require('fs'),
    os   = require('os'),
    path = require('path'),
    cnst = require('constants');

var rimraf     = require('rimraf'),
    rimrafSync = rimraf.sync;

/* HELPERS */

var RDWR_EXCL = cnst.O_CREAT | cnst.O_TRUNC | cnst.O_RDWR | cnst.O_EXCL;

var generateName = function(rawAffixes, defaultPrefix) {
  var affixes = parseAffixes(rawAffixes, defaultPrefix);
  var now = new Date();
  var name = [affixes.prefix,
              now.getYear(), now.getMonth(), now.getDate(),
              '-',
              process.pid,
              '-',
              (Math.random() * 0x100000000 + 1).toString(36),
              affixes.suffix].join('');
  return path.join(affixes.dir || exports.dir, name);
};

var parseAffixes = function(rawAffixes, defaultPrefix) {
  var affixes = {prefix: null, suffix: null};
  if(rawAffixes) {
    switch (typeof(rawAffixes)) {
    case 'string':
      affixes.prefix = rawAffixes;
      break;
    case 'object':
      affixes = rawAffixes;
      break;
    default:
      throw new Error("Unknown affix declaration: " + affixes);
    }
  } else {
    affixes.prefix = defaultPrefix;
  }
  return affixes;
};

/* -------------------------------------------------------------------------
 * Don't forget to call track() if you want file tracking and exit handlers!
 * -------------------------------------------------------------------------
 * When any temp file or directory is created, it is added to filesToDelete
 * or dirsToDelete. The first time any temp file is created, a listener is
 * added to remove all temp files and directories at exit.
 */
var tracking = false;
var track = function(value) {
  tracking = (value !== false);
  return module.exports; // chainable
};
var exitListenerAttached = false;
var filesToDelete = [];
var dirsToDelete = [];

function deleteFileOnExit(filePath) {
  if (!tracking) return false;
  attachExitListener();
  filesToDelete.push(filePath);
}

function deleteDirOnExit(dirPath) {
  if (!tracking) return false;
  attachExitListener();
  dirsToDelete.push(dirPath);
}

function attachExitListener() {
  if (!tracking) return false;
  if (!exitListenerAttached) {
    process.addListener('exit', cleanupSync);
    exitListenerAttached = true;
  }
}

function cleanupFilesSync() {
  if (!tracking) {
    return false;
  }
  var count = 0;
  var toDelete;
  while ((toDelete = filesToDelete.shift()) !== undefined) {
    rimrafSync(toDelete);
    count++;
  }
  return count;
}

function cleanupFiles(callback) {
  if (!tracking) {
    if (callback) {
      callback(new Error("not tracking"));
    }
    return;
  }
  var count = 0;
  var left = filesToDelete.length;
  if (!left) {
    if (callback) {
      callback(null, count);
    }
    return;
  }
  var toDelete;
  var rimrafCallback = function(err) {
    if (!left) {
      // Prevent processing if aborted
      return;
    }
    if (err) {
      // This shouldn't happen; pass error to callback and abort
      // processing
      if (callback) {
        callback(err);
      }
      left = 0;
      return;
    } else {
      count++;
    }
    left--;
    if (!left && callback) {
      callback(null, count);
    }
  };
  while ((toDelete = filesToDelete.shift()) !== undefined) {
    rimraf(toDelete, rimrafCallback);
  }
}

function cleanupDirsSync() {
  if (!tracking) {
    return false;
  }
  var count = 0;
  var toDelete;
  while ((toDelete = dirsToDelete.shift()) !== undefined) {
    rimrafSync(toDelete);
    count++;
  }
  return count;
}

function cleanupDirs(callback) {
  if (!tracking) {
    if (callback) {
      callback(new Error("not tracking"));
    }
    return;
  }
  var count = 0;
  var left = dirsToDelete.length;
  if (!left) {
    if (callback) {
      callback(null, count);
    }
    return;
  }
  var toDelete;
  var rimrafCallback = function (err) {
    if (!left) {
      // Prevent processing if aborted
      return;
    }
    if (err) {
      // rimraf handles most "normal" errors; pass the error to the
      // callback and abort processing
      if (callback) {
        callback(err, count);
      }
      left = 0;
      return;
    } else {
      count;
    }
    left--;
    if (!left && callback) {
      callback(null, count);
    }
  };
  while ((toDelete = dirsToDelete.shift()) !== undefined) {
    rimraf(toDelete, rimrafCallback);
  }
}

function cleanupSync() {
  if (!tracking) {
    return false;
  }
  var fileCount = cleanupFilesSync();
  var dirCount  = cleanupDirsSync();
  return {files: fileCount, dirs: dirCount};
}

function cleanup(callback) {
  if (!tracking) {
    if (callback) {
      callback(new Error("not tracking"));
    }
    return;
  }
  cleanupFiles(function(fileErr, fileCount) {
    if (fileErr) {
      if (callback) {
        callback(fileErr, {files: fileCount})
      }
    } else {
      cleanupDirs(function(dirErr, dirCount) {
        if (callback) {
          callback(dirErr, {files: fileCount, dirs: dirCount});
        }
      });
    }
  });
}

/* DIRECTORIES */

function mkdir(affixes, callback) {
  var dirPath = generateName(affixes, 'd-');
  fs.mkdir(dirPath, 0700, function(err) {
    if (!err) {
      deleteDirOnExit(dirPath);
    }
    if (callback) {
      callback(err, dirPath);
    }
  });
}

function mkdirSync(affixes) {
  var dirPath = generateName(affixes, 'd-');
  fs.mkdirSync(dirPath, 0700);
  deleteDirOnExit(dirPath);
  return dirPath;
}

/* FILES */

function open(affixes, callback) {
  var filePath = generateName(affixes, 'f-');
  fs.open(filePath, RDWR_EXCL, 0600, function(err, fd) {
    if (!err) {
      deleteFileOnExit(filePath);
    }
    if (callback) {
      callback(err, {path: filePath, fd: fd});
    }
  });
}

function openSync(affixes) {
  var filePath = generateName(affixes, 'f-');
  var fd = fs.openSync(filePath, RDWR_EXCL, 0600);
  deleteFileOnExit(filePath);
  return {path: filePath, fd: fd};
}

function createWriteStream(affixes) {
  var filePath = generateName(affixes, 's-');
  var stream = fs.createWriteStream(filePath, {flags: RDWR_EXCL, mode: 0600});
  deleteFileOnExit(filePath);
  return stream;
}

/* EXPORTS */
// Settings
exports.dir               = path.resolve(os.tmpDir());
exports.track             = track;
// Functions
exports.mkdir             = mkdir;
exports.mkdirSync         = mkdirSync;
exports.open              = open;
exports.openSync          = openSync;
exports.path              = generateName;
exports.cleanup           = cleanup;
exports.cleanupSync       = cleanupSync;
exports.createWriteStream = createWriteStream;
