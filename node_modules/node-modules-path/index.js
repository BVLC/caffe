'use strict';

var path  = require('path');
var fs  = require('fs');

// Due to the fact that the node community doesn't feel that
// the node_modules resolution algorithm should be public method
// we have copy pasta'd it here.
function nodeModulePaths(from) {
  // guarantee that 'from' is absolute.
  from = path.resolve(from);

  // note: this approach *only* works when the path is guaranteed
  // to be absolute.  Doing a fully-edge-case-correct path.split
  // that works on both Windows and Posix is non-trivial.
  var splitRe = process.platform === 'win32' ? /[\/\\]/ : /\//;
  var paths = [];
  var parts = from.split(splitRe);

  for (var tip = parts.length - 1; tip >= 0; tip--) {
    // don't search in .../node_modules/node_modules
    if (parts[tip] === 'node_modules') {
      continue;
    }

    var dir = parts.slice(0, tip + 1).concat('node_modules').join(path.sep);
    paths.push(dir);
  }

  return paths;
}

function isSubdirectoryOf(parentPath, possibleChildPath) {
  return possibleChildPath.length > parentPath.length &&
    possibleChildPath.indexOf(parentPath) === 0;
}

/**
  A utility function for determining what path an addon may be found at. Addons
  will only be resolved in the project's own `node_modules/` directory, they
  do not follow the standard node `require` logic that a standard
  `require('mode-module')` lookup would use, which finds the module according
  to the `NODE_PATH`.

  A description of node's lookup logic can be found here:

  https://nodejs.org/api/modules.html#modules_all_together

  Using this method to discover the correct location of project's `node_modules`
  directory allows addons to be looked up properly even when that `node_modules`
  directory is outside of the project's root.

  This method checks the env variable `EMBER_NODE_PATH`. If present, its value
  is used to determine the `node_modules` path.

  Possible use cases for this include caching
  the `node_modules/` directory outside of a source code checkout, and
  ensuring the same source code (shared over a network) can be used with
  different environments (Linux, OSX) where binary compatibility may not
  exist.

  For example, if you have a project in /projects/my-app and its `node_modules`
  directory is at /resource/node_modules, you would:

  ```
    # Node uses this as its search path for standard `require('module')` calls
    export NODE_PATH=/resource/node_modules

    # So that ember addon discovery looks here
    export EMBER_NODE_PATH=/resource/node_modules

    cd /projects/my-app && ember build
  ```

  @private
  @method nodeModulesPath
  @param  {String} context The starting directory to use to find the
                            node_modules path. This will usually be the
                            project's root
  @return {String} absolute path to the node_modules directory
 */
module.exports = function nodeModulesPath(context) {

  var nodePath = process.env.EMBER_NODE_PATH;
  var contextPath = path.resolve(context);

  if (nodePath) {
    var configuredPath = path.resolve(nodePath);

    // The contextPath is likely the project root, or possibly a subdirectory in
    // node_modules/ nested dependencies. If it is more specific than the
    // the configuredPath (i.e. it is a subdirectory of the configuredPath)
    // prefer the more specific contextPath.
    if (isSubdirectoryOf(configuredPath, contextPath)) {
      return path.resolve(contextPath, 'node_modules');
    } else {
      return path.resolve(nodePath);
    }
  } else {
    var paths = nodeModulePaths(contextPath);

    for (var i = 0, l = paths.length; i < l; i++) {
      var nodeModulePathUnderTest = paths[i];

      if (fs.existsSync(nodeModulePathUnderTest)) {
        return nodeModulePathUnderTest;
      }

    }

    return null;
  }
};
