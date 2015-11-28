var fs = require('fs')
var path = require('path')
var exec = require('child_process').exec
var async = require('async')
var fileExists = fs.exists || path.exists

exports.fileExists = fileExists

// Async function that tells whether the executable specified for said browser exists on the system
var browserExeExists = findableBy(fileExists)
exports.browserExeExists = browserExeExists

// Async function that tells whether an executable is findable by the `where` command on Windows
var findableByWhere = findableBy(where)
exports.findableByWhere = findableByWhere

// Async function that tells whether an executable is findable by the `which` command on Unix
var findableByWhich = findableBy(which)
exports.findableByWhich = findableByWhich

exports.findableByWhichOrModule = findableByTypeOrModule(which)
exports.findableByWhereOrModule = findableByTypeOrModule(where)

function findableBy(func, cb){
  return function(cb){
    var browser = this
    if (browser.exe instanceof Array)
      async.filter(browser.exe, func, function(exes){
        cb(exes.length > 0)
      })
    else
      func(browser.exe, cb)
  }
}

function findableByTypeOrModule(type) {
  return function(cb) {
    var browser = this
    findableByModule.call(browser, function(findable){
      if(findable){
        cb(findable)
      }else{
        findableBy(type).call(browser, cb)
      }
    })
  }
}

function resolveModule(exe){
  var nodeModule
  var nodeModuleBin
  try {
    nodeModule = require(exe)
    nodeModuleBin = nodeModule.path
  } catch(err) {
    nodeModule = null
    nodeModuleBin = null
  }
  return nodeModuleBin
}

function findableByModule(cb){
  var browser = this
  var moduleBin = resolveModule(browser.exe)
  if(moduleBin){
    browser.exe = moduleBin
  }
  cb(!!moduleBin)
}

exports.where = where
function where(exe, cb){
  exec('where ' + exe, function(err, exePath){
    cb(!!exePath)
  })
}

exports.which = which
function which(exe, cb){
  exec('which ' + exe, function(err, exePath){
    cb(!!exePath)
  })
}
