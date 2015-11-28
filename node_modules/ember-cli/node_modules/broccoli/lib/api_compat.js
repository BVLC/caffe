var fs = require('fs')
var RSVP = require('rsvp')
var quickTemp = require('quick-temp')
var mapSeries = require('promise-map-series')
var rimraf = require('rimraf')


// Wrap a new-style plugin to provide the .read API

exports.NewStyleTreeWrapper = NewStyleTreeWrapper
function NewStyleTreeWrapper (newStyleTree) {
  this.newStyleTree = newStyleTree
  this.description = newStyleTree.description ||
    (newStyleTree.constructor && newStyleTree.constructor.name) ||
    'NewStyleTreeWrapper'
}

NewStyleTreeWrapper.prototype.read = function (readTree) {
  var tree = this.newStyleTree

  quickTemp.makeOrReuse(tree, 'cachePath')
  quickTemp.makeOrReuse(tree, 'outputPath') // reuse to keep name across rebuilds
  rimraf.sync(tree.outputPath)
  fs.mkdirSync(tree.outputPath)

  if (!tree.inputTrees && !tree.inputTree) {
    throw new Error('No inputTree/inputTrees set on tree: ' + this.description)
  }
  if (tree.inputTree && tree.inputTrees) {
    throw new Error('Cannot have both inputTree and inputTrees: ' + this.description)
  }

  var inputTrees = tree.inputTrees || [tree.inputTree]
  return mapSeries(inputTrees, readTree)
    .then(function (inputPaths) {
      if (tree.inputTree) { // singular
        tree.inputPath = inputPaths[0]
      } else { // plural
        tree.inputPaths = inputPaths
      }
      return RSVP.resolve().then(function () {
        return tree.rebuild()
      }).then(function () {
        return tree.outputPath
      }, function (err) {
        // Pull in properties from broccoliInfo, and wipe properties that we
        // won't support under the new API
        delete err.treeDir
        var broccoliInfo = err.broccoliInfo || {}
        err.file = broccoliInfo.file
        err.line = broccoliInfo.firstLine
        err.column = broccoliInfo.firstColumn
        throw err
      })
    })
}

NewStyleTreeWrapper.prototype.cleanup = function () {
  quickTemp.remove(this.newStyleTree, 'outputPath')
  quickTemp.remove(this.newStyleTree, 'cachePath')
  if (this.newStyleTree.cleanup) {
    return this.newStyleTree.cleanup()
  }
}
