var builder = require('./builder')
exports.Builder = builder.Builder
exports.loadBrocfile = builder.loadBrocfile
exports.server = require('./server')
exports.getMiddleware = require('./middleware')
exports.Watcher = require('./watcher')
var cli = require('./cli')
exports.cli = cli

exports.makeTree = function makeTree (dir) {
  throw new Error('broccoli.makeTree has been removed in favor of string literals.\nUse "' + dir + '" instead of broccoli.makeTree("' + dir + '").')
}
exports.bowerTrees = function bowerTrees (dir) {
  throw new Error('broccoli.bowerTrees has been extracted into the broccoli-bower plugin.\nUse `var findBowerTrees = require(\'broccoli-bower\'); findBowerTrees() // => array of trees` instead.')
}
exports.MergedTree = function MergedTree () {
  throw new Error('broccoli.MergedTree has been extracted into the broccoli-merge-trees plugin.\n' +
    'Run "npm install --save-dev broccoli-merge-trees" and use it like so in Brocfile.js:\n' +
    'var mergeTrees = require(\'broccoli-merge-trees\');\n' +
    'var myMergedTree = mergeTrees([tree1, tree2, tree3], { overwrite: true });')
}
