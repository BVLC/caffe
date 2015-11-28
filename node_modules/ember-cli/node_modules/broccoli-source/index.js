'use strict'

exports.Directory = Directory
function Directory(directoryPath, watched, options) {
  if (typeof directoryPath !== 'string') throw new Error('Expected a path (string), got ' + directoryPath)
  this._directoryPath = directoryPath
  this._watched = !!watched

  options = options || {}
  this._name = options.name || (this.constructor && this.constructor.name) || 'Directory'
  this._annotation = options.annotation

  // Remember current call stack (minus "Error" line)
  this._instantiationStack = (new Error).stack.replace(/[^\n]*\n/, '')
}

Directory.prototype.__broccoliFeatures__ = Object.freeze({
  persistentOutputFlag: true,
  sourceDirectories: true
})

Directory.prototype.__broccoliGetInfo__ = function(builderFeatures) {
  if (builderFeatures == null) builderFeatures = { persistentOutputFlag: true, sourceDirectories: true }
  if (!builderFeatures.persistentOutputFlag || !builderFeatures.sourceDirectories) {
    throw new Error('Minimum builderFeatures required: { persistentOutputFlag: true, sourceDirectories: true }')
  }

  return {
    nodeType: 'source',
    sourceDirectory: this._directoryPath,
    watched: this._watched,
    instantiationStack: this._instantiationStack,
    name: this._name,
    annotation: this._annotation
  }
}

Directory.prototype.read = function(readTree) {
  // Go through same interface as real Broccoli builder, so we don't have
  // separate code paths

  var pluginInterface = this.__broccoliGetInfo__()

  if (pluginInterface.watched) {
    return readTree(pluginInterface.sourceDirectory)
  } else {
    return pluginInterface.sourceDirectory
  }
}

Directory.prototype.cleanup = function() {
}

exports.WatchedDir = WatchedDir
WatchedDir.prototype = Object.create(Directory.prototype)
WatchedDir.prototype.constructor = WatchedDir
function WatchedDir(directoryPath, options) {
  Directory.call(this, directoryPath, true, options)
}

exports.UnwatchedDir = UnwatchedDir
UnwatchedDir.prototype = Object.create(Directory.prototype)
UnwatchedDir.prototype.constructor = UnwatchedDir
function UnwatchedDir(directoryPath, options) {
  Directory.call(this, directoryPath, false, options)
}
