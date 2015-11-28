var fs = require('fs')
var path = require('path')
var crypto = require('crypto')
var mkdirp = require('mkdirp')
var glob = require('glob')

var isWindows = process.platform === 'win32'
var pathSep   = path.sep

var keysForTreeWarningPrinted = false

exports.hashTree = hashTree
// This function is used by the watcher. It makes the following guarantees:
//
// (1) It never throws an exception.
//
// (2) It does not miss changes. In other words, if after this function returns,
// any part of the directory hierarchy changes, a subsequent call must
// return a different hash.
//
// (1) and (2) hold even in the face of a constantly-changing file system.
function hashTree (fullPath) {
  return hashStrings(keysForTree(fullPath))
}

exports.keysForTree = keysForTree
function keysForTree (fullPath, initialRelativePath) {
  var relativePath   = initialRelativePath || '.'
  var stats
  var statKeys

  stats = fs.statSync(fullPath)

  var childKeys = []
  statKeys = ['stats', stats.mode]
  if (stats.isDirectory()) {
    var fileIdentity = stats.dev + '\x00' + stats.ino
    var entries
    try {
      entries = fs.readdirSync(fullPath).sort()
    } catch (err) {
      console.warn('Warning: Failed to read directory ' + fullPath)
      console.warn(err.stack)
      childKeys = ['readdir failed']
      // That's all there is to say about this directory.
    }
    if (entries != null) {
      for (var i = 0; i < entries.length; i++) {
        var keys
        try {
          keys = keysForTree(
            path.join(fullPath, entries[i]),
            path.join(relativePath, entries[i])
          )
        } catch (err) {
          if (!keysForTreeWarningPrinted) {
            console.warn('Warning: failed to stat ' + path.join(fullPath, entries[i]))
            keysForTreeWarningPrinted = true
          }
          // The child has probably ceased to exist since we called
          // `readdirSync`, or it is a broken symlink.
          keys = ['missing']
        }
        childKeys = childKeys.concat(keys)
      }
    }
  } else if (stats.isFile()) {
    statKeys.push(stats.mtime.getTime())
    statKeys.push(stats.size)
  } else {
    throw new Error(fullPath + ': Unexpected file type')
  }

  return ['path', relativePath]
    .concat(statKeys)
    .concat(childKeys)
}


exports.hashStats = hashStats
function hashStats (stats, path) {
  // Both stats and path can be null
  var keys = []
  if (stats != null) {
    keys.push(stats.mode, stats.size, stats.mtime.getTime())
  }
  if (path != null) {
    keys.push(path)
  }
  return hashStrings(keys)
}


exports.hashStrings = hashStrings
function hashStrings (strings) {
  var joinedStrings = strings.join('\x00')
  return crypto.createHash('md5').update(new Buffer(joinedStrings, 'utf8')).digest('hex')
}


// If src is a file, dest is a file name. If src is a directory, dest is the
// directory that the contents of src will be copied into.
//
// This function refuses to overwrite files, but accepts if directories exist
// already.
//
// This function dereferences symlinks.
//
// Note that unlike cp(1), we do not special-case if dest is an existing
// directory, because relying on things to exist when we're in the middle of
// assembling a new tree is too brittle.
//
// This function is deprecated in favor of
// https://github.com/broccolijs/node-copy-dereference
//
// copy-dereference differs from copyRecursivelySync in that it won't call
// mkdirp to create the target directory (or the parent directory of the
// target file), which makes it stricter: (1) It's not OK for the target
// directory to exist already, and (2) missing parent directories will not
// automatically be created.
exports.copyRecursivelySync = copyRecursivelySync
function copyRecursivelySync (src, dest, _mkdirp) {
  if (_mkdirp == null) _mkdirp = true
  // Note: We could try readdir'ing and catching ENOTDIR exceptions, but that
  // is 3x slower than stat'ing in the common case that we have a file.
  var srcStats = fs.statSync(src)
  if (srcStats.isDirectory()) {
    mkdirp.sync(dest)
    var entries = fs.readdirSync(src).sort()
    for (var i = 0; i < entries.length; i++) {
      // Set _mkdirp to false when recursing to avoid extra mkdirp calls.
      copyRecursivelySync(src + '/' + entries[i], dest + '/' + entries[i], false)
    }
  } else {
    if (_mkdirp) {
      mkdirp.sync(path.dirname(dest))
    }
    copyPreserveSync(src, dest, srcStats)
  }
}

// This function is deprecated in favor of
// https://github.com/broccolijs/node-copy-dereference
//
// srcStats is optional; use it as an optimization to avoid double stats
// This function refuses to overwrite files.
exports.copyPreserveSync = copyPreserveSync
function copyPreserveSync (src, dest, srcStats) {
  if (srcStats == null) srcStats = fs.statSync(src)
  if (srcStats.isFile()) {
    var content = fs.readFileSync(src)
    fs.writeFileSync(dest, content, { flag: 'wx' })
    fs.utimesSync(dest, srcStats.atime, srcStats.mtime)
  } else {
    throw new Error('Unexpected file type for ' + src)
  }
}

exports.linkRecursivelySync = linkRecursivelySync
function linkRecursivelySync () {
  throw new Error('linkRecursivelySync has been removed; use copyRecursivelySync instead (note: it does not overwrite)')
}

exports.linkAndOverwrite = linkAndOverwrite
function linkAndOverwrite () {
  throw new Error('linkAndOverwrite has been removed; use copyPreserveSync instead (note: it does not overwrite)')
}


exports.assertAbsolutePaths = assertAbsolutePaths
function assertAbsolutePaths (paths) {
  for (var i = 0; i < paths.length; i++) {
    if (paths[i][0] !== '/') {
      throw new Error('Path must be absolute: "' + paths[i] + '"')
    }
  }
}


// Multi-glob with reasonable defaults, so APIs all behave the same
exports.multiGlob = multiGlob
function multiGlob (globs, globOptions) {
  if (!Array.isArray(globs)) {
    throw new TypeError("multiGlob's first argument must be an array");
  }
  var options = {
    follow: true,
    nomount: true,
    strict: true
  }
  for (var key in globOptions) {
    if (globOptions.hasOwnProperty(key)) {
      options[key] = globOptions[key]
    }
  }

  var pathSet = {}
  var paths = []
  for (var i = 0; i < globs.length; i++) {
    if (options.nomount && globs[i][0] === '/') {
      throw new Error('Absolute paths not allowed (`nomount` is enabled): ' + globs[i])
    }
    var matches = glob.sync(globs[i], options)
    if (matches.length === 0) {
      throw new Error('Path or pattern "' + globs[i] + '" did not match any files')
    }
    for (var j = 0; j < matches.length; j++) {
      if (!pathSet[matches[j]]) {
        pathSet[matches[j]] = true
        paths.push(matches[j])
      }
    }
  }
  return paths
}


// This function is deprecated in favor of
// https://github.com/broccolijs/node-symlink-or-copy
exports.symlinkOrCopyPreserveSync = symlinkOrCopyPreserveSync
function symlinkOrCopyPreserveSync (sourcePath, destPath) {
  if (isWindows) {
    copyRecursivelySync(sourcePath, destPath)
  } else {
    if (fs.lstatSync(sourcePath).isSymbolicLink()) {
      // When we encounter symlinks, follow them. This prevents indirection
      // from growing out of control. Note: At the moment `realpath` on Node
      // is 70x slower than native: https://github.com/joyent/node/issues/7902
      sourcePath = fs.realpathSync(sourcePath)
    } else if (sourcePath[0] !== pathSep) {
      sourcePath = process.cwd() + pathSep + sourcePath
    }

    fs.symlinkSync(sourcePath, destPath)
  }
}
