'use strict';
var path         = require('path');
var root         = process.cwd();
var accessSync   = require('fs').accessSync;
var lstatSync    = require('fs').lstatSync;
var readlinkSync = require('fs').readlinkSync;

function existsSync(filepath, parent){
  var depth, link, linkError, linkRoot, relativeLink, resolvedLink, stats;
  var resolvedPath = path.resolve(filepath);
  try {
    stats = lstatSync(resolvedPath);
    // if symlink, check if target
    if (stats && stats.isSymbolicLink()) {
      link         = readlinkSync(resolvedPath);
      linkRoot     = path.dirname(resolvedPath);
      
      if (link && link.indexOf('..') !== -1) {
        // resolve relative path
        depth = pathDepth(link);
        relativeLink = path.relative(path.resolve(linkRoot, depth), path.basename(link));
        resolvedLink = path.resolve(linkRoot, relativeLink);
        
      } else {
        // assume root and resolve
        resolvedLink = path.resolve(root, link);
      }
      
      try {
        accessSync(path.dirname(resolvedLink));
        
      } catch (err) {
        if (err.code === "ENOENT") {
          // Log message for user so they can investigate
          console.log(err.message);
          console.log('Please verify that the symlink for ' + resolvedPath + ' can be resolved from ' + root + '.');
        }
      }
      
      if (parent && parent === resolvedLink) {
        linkError = new Error('Circular symlink detected: ' + resolvedPath + ' -> ' + resolvedLink);
        throw linkError;
      }
      return existsSync(resolvedLink, resolvedPath);
      
    }
    return true;
    
  } catch (err) {
    if (err.message.match(/Circular symlink detected/)) {
      throw err;
    }
    
    return checkError(err);
  }
}

function checkError(err) {
  return err && err.code === "ENOENT" ? false : true;
}

function pathDepth(filepath) {
  return new Array(filepath.split(path.sep).length).join('..' + path.sep);
}

module.exports = existsSync;