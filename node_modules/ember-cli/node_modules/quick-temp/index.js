var fs = require('fs')
var path = require('path')
var mktemp = require('mktemp')
var rimraf = require('rimraf')
var underscoreString = require('underscore.string')

exports.makeOrRemake = makeOrRemake
function makeOrRemake(obj, prop, className) {
  if (obj[prop] != null) {
    remove(obj, prop)
  }
  return obj[prop] = makeTmpDir(obj, prop, className)
}

exports.makeOrReuse = makeOrReuse
function makeOrReuse(obj, prop, className) {
  if (obj[prop] != null) {
    return obj[prop]
  }
  return obj[prop] = makeTmpDir(obj, prop, className)
}

exports.remove = remove
function remove(obj, prop) {
  if (obj[prop] != null) {
    rimraf.sync(obj[prop])
  }
  obj[prop] = null
}


function makeTmpDir(obj, prop, className) {
  findBaseDir()
  if (className == null) className = obj.constructor && obj.constructor.name
  var tmpDirName = prettyTmpDirName(className, prop)
  return mktemp.createDirSync(path.join(baseDir, tmpDirName))
}

var baseDir

function findBaseDir () {
  if (baseDir == null) {
    try {
      if (fs.statSync('tmp').isDirectory()) {
        baseDir = fs.realpathSync('tmp')
      }
    } catch (err) {
      if (err.code !== 'ENOENT') throw err
      // We could try other directories, but for now we just create ./tmp if
      // it doesn't exist
      fs.mkdirSync('tmp')
      baseDir = fs.realpathSync('tmp')
    }
  }
}

function cleanString (s) {
  return underscoreString.underscored(s || '')
    .replace(/[^a-z_]/g, '')
    .replace(/^_+/, '')
}

function prettyTmpDirName (className, prop) {
  var cleanClassName = cleanString(className)
  if (cleanClassName === 'object') cleanClassName = ''
  if (cleanClassName) cleanClassName += '-'
  var cleanPropertyName = cleanString(prop)
  return cleanClassName + cleanPropertyName + '-XXXXXXXX.tmp'
}
